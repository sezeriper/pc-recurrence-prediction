from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from pc_recurrence import __version__
from pc_recurrence.image_data.dicom import (
    DicomVolume,
    SeriesKey,
    inspect_patient,
    load_dicom_volume,
    series_sha256,
)
from pc_recurrence.image_data.workbook import load_image_workbook, select_image_workbook_rows
from pc_recurrence.io import sha256_file

from .artifacts import (
    EMBEDDING_SUMMARY_COLUMNS,
    create_run_directory,
    write_json,
    write_npz,
    write_summary,
)
from .constants import (
    MERLIN_EMBEDDING_DIMENSION,
    MERLIN_FILENAME,
    MERLIN_HU_RANGE,
    MERLIN_INPUT_SIZE,
    MERLIN_REPOSITORY,
    MERLIN_REVISION,
    SPECTRE_BACKBONE_FILENAME,
    SPECTRE_COMBINER_FILENAME,
    SPECTRE_CROP_SIZE,
    SPECTRE_EMBEDDING_DIMENSION,
    SPECTRE_HU_RANGE,
    SPECTRE_REPOSITORY,
    SPECTRE_REVISION,
    ImageEncoderName,
)
from .foundation_models import (
    FoundationModelArtifacts,
    acquire_foundation_model,
    encode_merlin,
    encode_spectre,
    load_foundation_runtime,
)
from .foundation_preprocessing import prepare_merlin_input, prepare_spectre_input


@dataclass(frozen=True)
class CtSeriesCase:
    patient_id: str
    patient_dir: Path
    study_uid: str
    series_uid: str


@dataclass(frozen=True)
class CaseEncoding:
    patient_id: str
    patient_embedding: np.ndarray
    patch_embeddings: np.ndarray
    patch_starts: np.ndarray
    valid_voxel_counts: np.ndarray
    summary: dict[str, Any]
    record: dict[str, Any]


class CtSeriesDiscovery(list[CtSeriesCase]):
    """Eligible curated series plus auditable rows excluded before model loading."""

    def __init__(self, cases: list[CtSeriesCase], skipped: list[dict[str, Any]]) -> None:
        super().__init__(cases)
        self.skipped = skipped


ProgressReporter = Callable[[str], None]


def _report(progress: ProgressReporter | None, message: str) -> None:
    """Send an immediately visible, user-oriented pipeline progress message."""
    if progress is not None:
        progress(f"[pc-image-embed] {message}")


def _skipped_discovery_row(patient_id: str, reason: str) -> dict[str, str]:
    return {"patient_id": patient_id, "status": "skipped", "reason": reason}


def discover_ct_series_cases(
    dicom_root: Path,
    workbook_path: Path,
    patients: set[str] | None = None,
    *,
    skip_unavailable: bool = True,
) -> CtSeriesDiscovery:
    if not dicom_root.is_dir():
        raise ValueError(f"curated DICOM directory does not exist: {dicom_root}")
    curation_manifest = dicom_root / "curation_manifest.json"
    if not curation_manifest.is_file():
        raise ValueError(f"curated DICOM directory has no curation_manifest.json: {dicom_root}")
    rows = select_image_workbook_rows(load_image_workbook(workbook_path), patients)
    cases: list[CtSeriesCase] = []
    skipped: list[dict[str, Any]] = []
    for row in rows:
        if row.dicom_folder is None:
            reason = "workbook has no DICOM folder mapping"
            if skip_unavailable:
                skipped.append(_skipped_discovery_row(row.patient_id, reason))
                continue
            raise ValueError(f"Patient {row.patient_id!r} has no DICOM folder mapping")
        patient_dir = dicom_root / row.dicom_folder
        if not patient_dir.is_dir():
            reason = "no curated CT series directory"
            if skip_unavailable:
                skipped.append(_skipped_discovery_row(row.patient_id, reason))
                continue
            raise ValueError(f"Patient {row.patient_id!r} has no curated CT series directory")
        inspection = inspect_patient(patient_dir, patient_id=row.patient_id)
        if (
            inspection.geometry_status != "eligible"
            or inspection.study_uid is None
            or inspection.series_uid is None
        ):
            reason = (
                "curated CT series is invalid: "
                f"{inspection.reason or inspection.geometry_status}"
            )
            if skip_unavailable:
                skipped.append(_skipped_discovery_row(row.patient_id, reason))
                continue
            raise ValueError(
                f"Patient {row.patient_id!r} curated CT series is invalid: "
                f"{inspection.reason or inspection.geometry_status}"
            )
        cases.append(
            CtSeriesCase(
                patient_id=row.patient_id,
                patient_dir=patient_dir,
                study_uid=inspection.study_uid,
                series_uid=inspection.series_uid,
            )
        )
    if not cases and not skip_unavailable:
        raise ValueError("the selected workbook cohort contains no curated CT series")
    return CtSeriesDiscovery(cases, skipped)


def _safe_state_name(patient_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in patient_id)


def _state_paths(state_dir: Path, patient_id: str) -> tuple[Path, Path]:
    stem = _safe_state_name(patient_id)
    return state_dir / f"{stem}.json", state_dir / f"{stem}.npz"


def _load_cached_case(
    state_json: Path,
    state_npz: Path,
    *,
    ct_series_sha256: str,
    encoder_name: ImageEncoderName,
    model_fingerprint: str,
) -> CaseEncoding | None:
    if not state_json.is_file() or not state_npz.is_file():
        return None
    metadata = json.loads(state_json.read_text(encoding="utf-8"))
    if (
        metadata.get("ct_series_sha256") != ct_series_sha256
        or metadata.get("encoder") != encoder_name.value
        or metadata.get("model_fingerprint") != model_fingerprint
    ):
        return None
    with np.load(state_npz, allow_pickle=False) as cached:
        return CaseEncoding(
            patient_id=metadata["patient_id"],
            patient_embedding=cached["patient_embedding"].astype(np.float32),
            patch_embeddings=cached["patch_embeddings"].astype(np.float32),
            patch_starts=cached["patch_starts"].astype(np.int32),
            valid_voxel_counts=cached["valid_voxel_counts"].astype(np.int64),
            summary=metadata["summary"],
            record=metadata["record"],
        )


def _encode_foundation_case(
    encoder: Any,
    case: CtSeriesCase,
    volume: DicomVolume,
    ct_series_digest: str,
    encoder_name: ImageEncoderName,
) -> CaseEncoding:
    image = nib.Nifti1Image(volume.volume_hu, volume.affine_ras)
    native_shape = tuple(int(value) for value in image.shape)
    started = time.perf_counter()
    if encoder_name is ImageEncoderName.SPECTRE:
        prepared = prepare_spectre_input(image)
        tokens, _ = encode_spectre(
            encoder, prepared.data, expected_grid=prepared.grid_size
        )
        patient_embedding = tokens[0]
        patch_embeddings = tokens[1:]
        dimension = SPECTRE_EMBEDDING_DIMENSION
        valid_counts = np.full(
            patch_embeddings.shape[0], prepared.valid_voxel_count, dtype=np.int64
        )
    elif encoder_name is ImageEncoderName.MERLIN:
        prepared = prepare_merlin_input(image)
        patient_embedding = encode_merlin(encoder, prepared.data)
        patch_embeddings = patient_embedding[None]
        dimension = MERLIN_EMBEDDING_DIMENSION
        valid_counts = np.asarray([prepared.valid_voxel_count], dtype=np.int64)
    inference_seconds = time.perf_counter() - started
    patient_embedding = np.asarray(patient_embedding, dtype=np.float32)
    patch_embeddings = np.asarray(patch_embeddings, dtype=np.float32)
    if patient_embedding.shape != (dimension,) or patch_embeddings.shape[1] != dimension:
        raise RuntimeError(
            f"unexpected {encoder_name.value} embedding shapes: "
            f"{patient_embedding.shape}, {patch_embeddings.shape}"
        )
    summary = {
        "patient_id": case.patient_id,
        "status": "embedded",
        "reason": None,
        "encoder": encoder_name.value,
        "native_ct_shape": native_shape,
        "resampled_shape": prepared.metadata.get("resampled_shape"),
        "patch_count": int(patch_embeddings.shape[0]),
        "embedding_dimension": dimension,
        "inference_seconds": round(inference_seconds, 3),
        "ct_series_sha256": ct_series_digest,
    }
    record = {
        **summary,
        "study_uid": volume.study_uid,
        "series_uid": volume.series_uid,
        "selected_instance_numbers": list(volume.selected_instance_numbers),
        "selected_sop_instance_uids": list(volume.selected_sop_instance_uids),
        "native_affine": np.asarray(image.affine).tolist(),
        "grid_size": list(prepared.grid_size),
        "patch_starts": prepared.patch_starts.tolist(),
        "valid_voxel_counts": valid_counts.tolist(),
        "preprocessing": prepared.metadata,
    }
    return CaseEncoding(
        case.patient_id,
        patient_embedding,
        patch_embeddings,
        prepared.patch_starts,
        valid_counts,
        summary,
        record,
    )


def _foundation_manifest(
    encoder_name: ImageEncoderName,
    artifacts: FoundationModelArtifacts,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    centering_text = "CT volume center"
    if encoder_name is ImageEncoderName.SPECTRE:
        return (
            {
                "repository": SPECTRE_REPOSITORY,
                "revision": SPECTRE_REVISION,
                "architecture": "SPECTRE-Large ViT plus feature combiner",
                "files": [
                    {"filename": SPECTRE_BACKBONE_FILENAME, "sha256": artifacts.hashes[0]},
                    {"filename": SPECTRE_COMBINER_FILENAME, "sha256": artifacts.hashes[1]},
                ],
                "weights_license": "CC-BY-NC-SA; non-commercial research use",
            },
            {
                "capture": "feature-combiner CLS scan token",
                "embedding_dimension": SPECTRE_EMBEDDING_DIMENSION,
                "l2_normalization": False,
                "inference_precision": "float16 autocast with float32 saved embeddings",
            },
            {
                "orientation": "RAS",
                "spacing": "native",
                "hu_clip": list(SPECTRE_HU_RANGE),
                "crop_size": list(SPECTRE_CROP_SIZE),
                "centering": centering_text,
                "padding": "-1000 HU outside the selected CT volume",
            },
            SPECTRE_EMBEDDING_DIMENSION,
        )
    return (
        {
            "repository": MERLIN_REPOSITORY,
            "revision": MERLIN_REVISION,
            "architecture": "Merlin inflated ResNet-152 image encoder",
            "files": [{"filename": MERLIN_FILENAME, "sha256": artifacts.hashes[0]}],
            "unused_text_tower_loaded": False,
        },
        {
            "capture": "global average-pooled image encoder output",
            "embedding_dimension": MERLIN_EMBEDDING_DIMENSION,
            "l2_normalization": False,
            "inference_precision": "float16 autocast with float32 saved embeddings",
        },
        {
            "orientation": "RAS",
            "spacing_mm": [1.5, 1.5, 3.0],
            "hu_clip": list(MERLIN_HU_RANGE),
            "intensity_range": [0.0, 1.0],
            "input_size": list(MERLIN_INPUT_SIZE),
            "centering": centering_text,
            "padding": "-1000 HU before intensity scaling",
        },
        MERLIN_EMBEDDING_DIMENSION,
    )


def run_embedding(
    dicom_root: Path,
    output_root: Path,
    model_cache: Path,
    *,
    workbook_path: Path,
    encoder_name: ImageEncoderName = ImageEncoderName.SPECTRE,
    run_dir: Path | None = None,
    patients: set[str] | None = None,
    resume: bool = True,
    force: bool = False,
    local_model_only: bool = False,
    skip_unavailable: bool = True,
    progress: ProgressReporter | None = None,
) -> Path:
    _report(progress, f"Inspecting curated CT series in {dicom_root}.")
    discovery = (
        discover_ct_series_cases(
            dicom_root,
            workbook_path,
            patients,
            skip_unavailable=True,
        )
        if skip_unavailable
        else discover_ct_series_cases(dicom_root, workbook_path, patients)
    )
    cases = list(discovery)
    discovery_failures = list(getattr(discovery, "skipped", []))
    _report(
        progress,
        f"Found {len(cases)} eligible CT series; "
        f"{len(discovery_failures)} patient(s) will be skipped.",
    )
    for failure in discovery_failures:
        _report(
            progress,
            f"Skipping {failure['patient_id']}: {failure['reason']}.",
        )
    destination = run_dir or create_run_directory(output_root / encoder_name.value)
    destination.mkdir(parents=True, exist_ok=True)
    state_dir = destination / ".state"
    if force and state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(exist_ok=True)

    _report(
        progress,
        f"Acquiring the pinned {encoder_name.value.upper()} checkpoint "
        "(downloads only when it is not already cached).",
    )
    foundation_artifacts = acquire_foundation_model(
        encoder_name, model_cache, local_files_only=local_model_only
    )
    _report(progress, "Checkpoint verified; loading the model and checking the runtime.")
    encoder, runtime = load_foundation_runtime(encoder_name, foundation_artifacts)
    _report(
        progress,
        f"Model ready on {runtime.device_type}: {runtime.device_name} "
        f"(startup check: {runtime.smoke_test_seconds:.1f}s).",
    )
    model_fingerprint = ":".join(foundation_artifacts.hashes)
    model_manifest, feature_manifest, preprocessing_manifest, dimension = _foundation_manifest(
        encoder_name, foundation_artifacts
    )

    completed: list[CaseEncoding] = []
    rows: list[dict[str, Any]] = [
        {**failure, "encoder": encoder_name.value} for failure in discovery_failures
    ]
    records: list[dict[str, Any]] = rows.copy()
    failures: list[dict[str, Any]] = rows.copy()
    started_run = time.perf_counter()
    total_cases = len(cases)
    for case_index, case in enumerate(cases, start=1):
        progress_label = f"[{case_index}/{total_cases}] {case.patient_id}"
        _report(progress, f"{progress_label}: loading the selected CT series.")
        volume = load_dicom_volume(
            case.patient_dir,
            patient_id=case.patient_id,
            selection=SeriesKey(case.study_uid, case.series_uid),
        )
        ct_series_digest = series_sha256(volume.files)
        state_json, state_npz = _state_paths(state_dir, case.patient_id)
        cached = (
            _load_cached_case(
                state_json,
                state_npz,
                ct_series_sha256=ct_series_digest,
                encoder_name=encoder_name,
                model_fingerprint=model_fingerprint,
            )
            if resume and not force
            else None
        )
        if cached is not None:
            completed.append(cached)
            rows.append(cached.summary)
            records.append(cached.record)
            _report(
                progress,
                f"{progress_label}: reused the resumable cache "
                f"({cached.summary['patch_count']} patches).",
            )
            continue
        try:
            _report(progress, f"{progress_label}: encoding.")
            encoded = _encode_foundation_case(
                encoder, case, volume, ct_series_digest, encoder_name
            )
            completed.append(encoded)
            rows.append(encoded.summary)
            records.append(encoded.record)
            write_npz(
                state_npz,
                patient_embedding=encoded.patient_embedding,
                patch_embeddings=encoded.patch_embeddings,
                patch_starts=encoded.patch_starts,
                valid_voxel_counts=encoded.valid_voxel_counts,
            )
            write_json(
                {
                    "patient_id": case.patient_id,
                    "ct_series_sha256": ct_series_digest,
                    "encoder": encoder_name.value,
                    "model_fingerprint": model_fingerprint,
                    "summary": encoded.summary,
                    "record": encoded.record,
                },
                state_json,
            )
            _report(
                progress,
                f"{progress_label}: complete — {encoded.summary['patch_count']} patches "
                f"in {encoded.summary['inference_seconds']:.1f}s.",
            )
        except Exception as error:
            failure = {
                "patient_id": case.patient_id,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                "encoder": encoder_name.value,
                "ct_series_sha256": ct_series_digest,
            }
            rows.append(failure)
            records.append(failure)
            failures.append(failure)
            _report(progress, f"{progress_label}: failed — {failure['reason']}.")

    if completed:
        patient_ids = np.asarray([item.patient_id for item in completed], dtype=np.str_)
        patient_embeddings = np.stack([item.patient_embedding for item in completed]).astype(
            np.float32
        )
        patch_patient_ids = np.concatenate(
            [np.repeat(item.patient_id, item.patch_embeddings.shape[0]) for item in completed]
        ).astype(np.str_)
        patch_indices = np.concatenate(
            [np.arange(item.patch_embeddings.shape[0], dtype=np.int32) for item in completed]
        )
        patch_embeddings = np.concatenate(
            [item.patch_embeddings for item in completed], axis=0
        ).astype(np.float32)
        patch_starts = np.concatenate([item.patch_starts for item in completed], axis=0)
        valid_counts = np.concatenate([item.valid_voxel_counts for item in completed], axis=0)
    else:
        patient_ids = np.asarray([], dtype=np.str_)
        patient_embeddings = np.empty((0, dimension), dtype=np.float32)
        patch_patient_ids = np.asarray([], dtype=np.str_)
        patch_indices = np.asarray([], dtype=np.int32)
        patch_embeddings = np.empty((0, dimension), dtype=np.float32)
        patch_starts = np.empty((0, 3), dtype=np.int32)
        valid_counts = np.asarray([], dtype=np.int64)

    write_npz(
        destination / "image_embeddings.npz",
        patient_ids=patient_ids,
        embeddings=patient_embeddings,
        patch_counts=np.asarray(
            [item.patch_embeddings.shape[0] for item in completed], dtype=np.int32
        ),
        encoder=np.asarray(encoder_name.value),
    )
    write_npz(
        destination / "patch_embeddings.npz",
        patient_ids=patch_patient_ids,
        patch_indices=patch_indices,
        patch_starts=patch_starts,
        valid_voxel_counts=valid_counts,
        embeddings=patch_embeddings,
    )
    write_summary(rows, destination / "embedding_summary.csv", EMBEDDING_SUMMARY_COLUMNS)
    source_manifest = dicom_root / "curation_manifest.json"
    status_counts = dict(Counter(row["status"] for row in rows))
    elapsed_seconds = time.perf_counter() - started_run
    write_json(
        {
            "pipeline_version": __version__,
            "stage": f"{encoder_name.value}_selected_ct_embedding",
            "encoder": encoder_name.value,
            "status": "complete" if not failures else "completed_with_failures",
            "dicom_root": str(dicom_root.resolve()),
            "workbook": str(workbook_path.resolve()),
            "source_curation_manifest_sha256": sha256_file(source_manifest),
            "source_kind": "workbook-selected CT series range",
            "skip_unavailable": skip_unavailable,
            "model": model_manifest,
            "feature_extraction": feature_manifest,
            "preprocessing": preprocessing_manifest,
            "runtime": runtime.to_dict(),
            "patient_count": len(completed),
            "patch_count": int(patch_embeddings.shape[0]),
            "status_counts": status_counts,
            "patients": records,
            "failures": failures,
            "elapsed_seconds": elapsed_seconds,
            "artifacts": {
                "patient_embeddings": "image_embeddings.npz",
                "patch_embeddings": "patch_embeddings.npz",
                "summary": "embedding_summary.csv",
            },
            "provisional_research_output": True,
        },
        destination / "run_manifest.json",
    )
    _report(
        progress,
        "Finished: "
        f"{len(completed)} embedded, "
        f"{status_counts.get('skipped', 0)} skipped, "
        f"{status_counts.get('failed', 0)} failed in {elapsed_seconds:.1f}s. "
        f"Artifacts: {destination}",
    )
    return destination
