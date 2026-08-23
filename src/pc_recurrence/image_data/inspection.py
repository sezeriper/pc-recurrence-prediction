from __future__ import annotations

from pathlib import Path
from typing import Any

from pc_recurrence import __version__
from pc_recurrence.io import write_json, write_summary

from .dicom import SeriesInspection, inspect_patient, patient_directories
from .workbook import load_image_workbook

INSPECTION_SUMMARY_COLUMNS = (
    "patient_id",
    "status",
    "reason",
    "study_uid",
    "series_uid",
    "dicom_file_count",
    "native_shape",
    "native_spacing_mm",
    "maximum_slice_gap_mm",
    "phase_status",
)


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
        inspect_patient(path, patient_id=patient_ids.get(path.name, path.name))
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
    }


def write_inspection_run(
    inspections: list[SeriesInspection],
    run_dir: Path,
    dicom_root: Path,
    workbook_path: Path | None = None,
) -> tuple[Path, Path]:
    rows = [_inspection_row(item) for item in inspections]
    summary = write_summary(rows, run_dir / "inspection_summary.csv", INSPECTION_SUMMARY_COLUMNS)
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
