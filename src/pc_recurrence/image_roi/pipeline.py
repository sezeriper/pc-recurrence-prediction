from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

from pc_recurrence import __version__

from .artifacts import (
    SEGMENTATION_SUMMARY_COLUMNS,
    create_run_directory,
    read_json,
    save_case_artifacts,
    write_json,
    write_summary,
)
from .constants import (
    BUNDLE_REPOSITORY,
    BUNDLE_VERSION,
    HU_RANGE,
    MAX_PANCREAS_DISTANCE_MM,
    MIN_TUMOR_VOLUME_MM3,
    MODEL_REVISION,
    MODEL_SHA256,
    ROI_MARGIN_MM,
    ROI_SIZE,
    ROI_TARGET,
    SW_BATCH_SIZE,
    SW_OVERLAP,
    TARGET_SPACING_MM,
)
from .dicom import (
    SeriesInspection,
    SeriesKey,
    inspect_patient,
    load_dicom_volume,
    patient_directories,
    series_sha256,
)
from .model import acquire_model, segment_volume, validate_and_load_runtime
from .processing import expanded_bounding_box, select_tumor_component, voxel_volume_mm3
from .workbook import load_image_workbook


def inspect_dataset(
    dicom_root: Path,
    patients: set[str] | None = None,
    workbook_path: Path | None = None,
) -> list[SeriesInspection]:
    directories = patient_directories(dicom_root)
    if patients is not None:
        directories = [path for path in directories if path.name in patients]
    patient_ids = (
        {
            row.dicom_folder: row.patient_id
            for row in load_image_workbook(workbook_path)
            if row.dicom_folder is not None
        }
        if workbook_path
        else {}
    )
    return [
        inspect_patient(
            path,
            patient_id=patient_ids.get(path.name, path.name),
        )
        for path in directories
    ]


def _inspection_row(inspection: SeriesInspection) -> dict[str, Any]:
    return {
        "patient_id": inspection.patient_id,
        "status": inspection.geometry_status,
        "reason": inspection.reason,
        "study_uid": inspection.study_uid,
        "series_uid": inspection.series_uid,
        "dicom_file_count": inspection.file_count,
        "native_shape": inspection.shape,
        "native_spacing_mm": inspection.spacing_mm,
        "maximum_slice_gap_mm": inspection.maximum_slice_gap_mm,
        "phase_status": inspection.phase_status,
        "preprocessed_shape": None,
        "patch_count": None,
        "inference_seconds": None,
        "roi_volume_mm3": None,
        "patient_artifact_dir": None,
    }


def write_inspection_run(
    inspections: list[SeriesInspection],
    run_dir: Path,
    dicom_root: Path,
    workbook_path: Path | None = None,
) -> tuple[Path, Path]:
    rows = [_inspection_row(item) for item in inspections]
    summary = write_summary(
        rows, run_dir / "segmentation_summary.csv", SEGMENTATION_SUMMARY_COLUMNS
    )
    manifest = write_json(
        {
            "pipeline_version": __version__,
            "stage": "dicom_inspection",
            "status": "complete",
            "dicom_root": str(dicom_root.resolve()),
            "workbook": str(workbook_path.resolve()) if workbook_path else None,
            "patient_count": len(inspections),
            "patients": [item.to_dict() for item in inspections],
            "phase_policy": "phase_unverified",
            "failures": [row for row in rows if row["status"] != "eligible"],
        },
        run_dir / "run_manifest.json",
    )
    return summary, manifest


def _resume_state(
    state_path: Path,
    patient_dir: Path,
    selection: SeriesKey,
    selected_sop_instance_uids: tuple[str, ...],
) -> dict[str, Any] | None:
    if not state_path.exists():
        return None
    state = read_json(state_path)
    cached_inspection = state.get("record", {}).get("inspection", {})
    if (
        cached_inspection.get("study_uid") != selection.study_uid
        or cached_inspection.get("series_uid") != selection.series_uid
        or cached_inspection.get("selected_sop_instance_uids", [])
        != list(selected_sop_instance_uids)
    ):
        return None
    if state.get("status") == "detected" and not patient_dir.exists():
        return None
    return state


def run_segmentation(
    dicom_root: Path,
    output_root: Path,
    model_cache: Path,
    *,
    workbook_path: Path | None = None,
    run_dir: Path | None = None,
    patients: set[str] | None = None,
    resume: bool = True,
    force: bool = False,
    local_model_only: bool = False,
) -> Path:
    destination = run_dir or create_run_directory(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    state_dir = destination / ".state"
    state_dir.mkdir(exist_ok=True)
    inspections = inspect_dataset(dicom_root, patients, workbook_path)
    if not inspections:
        raise ValueError("no matching patient directories were found")

    model_path = acquire_model(model_cache, local_files_only=local_model_only)
    model, runtime = validate_and_load_runtime(model_path)
    rows: list[dict[str, Any]] = []
    patient_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for inspection in inspections:
        state_path = state_dir / f"{inspection.patient_id}.json"
        patient_output = destination / inspection.patient_id
        selection_key = (
            SeriesKey(inspection.study_uid, inspection.series_uid)
            if inspection.study_uid is not None and inspection.series_uid is not None
            else None
        )
        if force:
            if patient_output.exists():
                shutil.rmtree(patient_output)
            state_path.unlink(missing_ok=True)
        if resume and not force and selection_key is not None:
            cached = _resume_state(
                state_path,
                patient_output,
                selection_key,
                inspection.selected_sop_instance_uids,
            )
            if cached is not None:
                rows.append(cached["summary"])
                patient_records.append(cached["record"])
                if cached["summary"]["status"] == "failed":
                    failures.append(cached["record"])
                continue
            if patient_output.exists():
                shutil.rmtree(patient_output)

        summary = _inspection_row(inspection)
        summary["roi_target"] = ROI_TARGET
        record: dict[str, Any] = {
            "inspection": inspection.to_dict(),
            "roi_target": ROI_TARGET,
        }
        if inspection.geometry_status != "eligible":
            record["status"] = inspection.geometry_status
            write_json(
                {
                    "status": inspection.geometry_status,
                    "roi_target": ROI_TARGET,
                    "summary": summary,
                    "record": record,
                },
                state_path,
            )
            rows.append(summary)
            patient_records.append(record)
            continue
        assert selection_key is not None

        try:
            started = time.perf_counter()
            volume = load_dicom_volume(
                inspection.patient_dir,
                patient_id=inspection.patient_id,
                selection=selection_key,
            )
            dicom_checksum = series_sha256(volume.files)
            segmentation = segment_volume(model, volume.volume_hu, volume.affine_ras)
            selection = select_tumor_component(segmentation.labels_original, volume.affine_ras)
            target_mask = selection.pancreas_mask
            roi_volume = float(target_mask.sum()) * voxel_volume_mm3(volume.affine_ras)
            summary.update(
                {
                    "native_shape": volume.shape,
                    "native_spacing_mm": volume.spacing_mm,
                    "maximum_slice_gap_mm": volume.maximum_slice_gap_mm,
                    "preprocessed_shape": segmentation.preprocessed_shape,
                    "patch_count": segmentation.patch_count,
                    "inference_seconds": round(segmentation.inference_seconds, 3),
                    "roi_volume_mm3": roi_volume if target_mask.any() else None,
                }
            )
            record.update(
                {
                    "dicom_sha256": dicom_checksum,
                    "selected_instance_numbers": list(volume.selected_instance_numbers),
                    "selected_sop_instance_uids": list(volume.selected_sop_instance_uids),
                    "predicted_pancreas_voxels": int(selection.pancreas_mask.sum()),
                    "predicted_raw_tumor_voxels": int((segmentation.labels_original == 2).sum()),
                    "selected_tumor_voxels": int(selection.tumor_mask.sum()),
                    "selected_tumor_volume_mm3": selection.selected_volume_mm3,
                    "component_decisions": [decision.to_dict() for decision in selection.decisions],
                    "preprocessed_shape": list(segmentation.preprocessed_shape),
                    "patch_count": segmentation.patch_count,
                    "inference_seconds": segmentation.inference_seconds,
                }
            )
            if not target_mask.any():
                summary["status"] = "no_pancreas"
                summary["reason"] = "no qualifying pancreas mask"
                record["status"] = "no_pancreas"
            else:
                bbox = expanded_bounding_box(target_mask, volume.affine_ras)
                bbox_metadata = {
                    **bbox.to_dict(),
                    "patient_id": volume.patient_id,
                    "study_uid": volume.study_uid,
                    "series_uid": volume.series_uid,
                    "selected_component": selection.selected_component,
                    "roi_target": ROI_TARGET,
                    "roi_volume_mm3": roi_volume,
                    "phase_status": volume.phase_status,
                    "provisional_research_output": True,
                }
                artifacts = save_case_artifacts(
                    patient_output,
                    volume.volume_hu,
                    volume.affine_ras,
                    selection.pancreas_mask,
                    bbox,
                    bbox_metadata,
                )
                summary["status"] = "detected"
                summary["reason"] = None
                summary["patient_artifact_dir"] = inspection.patient_id
                record.update(
                    {
                        "status": "detected",
                        "bbox": bbox_metadata,
                        "artifacts": artifacts,
                    }
                )
            record["total_patient_seconds"] = time.perf_counter() - started
        except Exception as exc:
            summary["status"] = "failed"
            summary["reason"] = f"{type(exc).__name__}: {exc}"
            record.update({"status": "failed", "error": summary["reason"]})
            failures.append(record)
        write_json(
            {
                "status": summary["status"],
                "roi_target": ROI_TARGET,
                "summary": summary,
                "record": record,
            },
            state_path,
        )
        rows.append(summary)
        patient_records.append(record)

    write_summary(rows, destination / "segmentation_summary.csv", SEGMENTATION_SUMMARY_COLUMNS)
    write_json(
        {
            "pipeline_version": __version__,
            "stage": "pancreas_tumor_segmentation_roi",
            "status": "complete_with_failures" if failures else "complete",
            "provisional_research_output": True,
            "diagnostic_use": False,
            "dicom_root": str(dicom_root.resolve()),
            "workbook": str(workbook_path.resolve()) if workbook_path else None,
            "model": {
                "repository": BUNDLE_REPOSITORY,
                "bundle_version": BUNDLE_VERSION,
                "revision": MODEL_REVISION,
                "model_sha256": MODEL_SHA256,
                "path": str(model_path.resolve()),
            },
            "runtime": runtime.to_dict(),
            "preprocessing": {
                "orientation": "RAS",
                "spacing_mm": list(TARGET_SPACING_MM),
                "hu_clip": list(HU_RANGE),
                "intensity_scale": [0.0, 1.0],
            },
            "inference": {
                "precision": "float32",
                "automatic_mixed_precision": False,
                "roi_size": list(ROI_SIZE),
                "sw_batch_size": SW_BATCH_SIZE,
                "overlap": SW_OVERLAP,
                "blending": "constant",
                "stitching_device": "cpu",
                "inference_device": runtime.device_type,
            },
            "tumor_filtering": {
                "connectivity": 26,
                "minimum_volume_mm3": MIN_TUMOR_VOLUME_MM3,
                "maximum_pancreas_distance_mm": MAX_PANCREAS_DISTANCE_MM,
                "selection": "largest qualifying component",
            },
            "roi_margin_mm": ROI_MARGIN_MM,
            "roi_target": ROI_TARGET,
            "patient_count": len(rows),
            "status_counts": {
                status: sum(row["status"] == status for row in rows)
                for status in sorted({str(row["status"]) for row in rows})
            },
            "patients": patient_records,
            "failures": failures,
            "artifacts": {
                "summary": "segmentation_summary.csv",
                "patient_outputs": "detected cases only",
            },
        },
        destination / "run_manifest.json",
    )
    return destination
