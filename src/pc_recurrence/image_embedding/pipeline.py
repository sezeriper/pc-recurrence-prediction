from __future__ import annotations

import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from pc_recurrence import __version__

from .artifacts import create_run_directory, write_json, write_npz, write_summary
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
    sha256_file,
)
from .foundation_preprocessing import prepare_merlin_input, prepare_spectre_input


@dataclass(frozen=True)
class RoiCase:
    patient_id: str
    patient_dir: Path
    roi_ct_path: Path
    roi_pancreas_mask_path: Path | None
    roi_target: str | None


@dataclass(frozen=True)
class CaseEncoding:
    patient_id: str
    patient_embedding: np.ndarray
    patch_embeddings: np.ndarray
    patch_starts: np.ndarray
    valid_voxel_counts: np.ndarray
    summary: dict[str, Any]
    record: dict[str, Any]


def _patient_sort_key(value: str) -> tuple[str, int, str]:
    prefix, separator, suffix = value.rpartition(" ")
    if separator and suffix.isdigit():
        return prefix.casefold(), int(suffix), value.casefold()
    return value.casefold(), -1, value.casefold()


def discover_roi_cases(roi_run: Path, patients: set[str] | None = None) -> list[RoiCase]:
    if not roi_run.is_dir():
        raise ValueError(f"ROI run directory does not exist: {roi_run}")
    manifest_path = roi_run / "run_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"ROI run has no run_manifest.json: {roi_run}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    roi_target = manifest.get("roi_target")
    cases: list[RoiCase] = []
    for path in roi_run.iterdir():
        roi_ct = path / "roi_ct.nii.gz"
        mask = path / "roi_pancreas_mask.nii.gz"
        if path.is_dir() and roi_ct.is_file() and (patients is None or path.name in patients):
            cases.append(
                RoiCase(
                    path.name,
                    path,
                    roi_ct,
                    mask if mask.is_file() else None,
                    roi_target,
                )
            )
    cases.sort(key=lambda case: _patient_sort_key(case.patient_id))
    if patients is not None:
        found = {case.patient_id for case in cases}
        missing = sorted(patients - found, key=_patient_sort_key)
        if missing:
            raise ValueError(f"requested patients have no ROI CT artifact: {', '.join(missing)}")
    if not cases:
        raise ValueError("the selected ROI run contains no roi_ct.nii.gz artifacts")
    return cases


def _safe_state_name(patient_id: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in patient_id)


def _state_paths(state_dir: Path, patient_id: str) -> tuple[Path, Path]:
    stem = _safe_state_name(patient_id)
    return state_dir / f"{stem}.json", state_dir / f"{stem}.npz"


def _load_cached_case(
    state_json: Path,
    state_npz: Path,
    *,
    roi_sha256: str,
    mask_sha256: str | None,
    encoder_name: ImageEncoderName,
    model_fingerprint: str,
) -> CaseEncoding | None:
    if not state_json.is_file() or not state_npz.is_file():
        return None
    metadata = json.loads(state_json.read_text(encoding="utf-8"))
    if (
        metadata.get("roi_ct_sha256") != roi_sha256
        or metadata.get("roi_pancreas_mask_sha256") != mask_sha256
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
    case: RoiCase,
    encoder_name: ImageEncoderName,
) -> CaseEncoding:
    if case.roi_pancreas_mask_path is None:
        raise ValueError("strict foundation encoding requires roi_pancreas_mask.nii.gz")
    image = nib.load(case.roi_ct_path)
    mask = nib.load(case.roi_pancreas_mask_path)
    roi_sha256 = sha256_file(case.roi_ct_path)
    mask_sha256 = sha256_file(case.roi_pancreas_mask_path)
    native_shape = tuple(int(value) for value in image.shape)
    started = time.perf_counter()
    if encoder_name is ImageEncoderName.SPECTRE:
        prepared = prepare_spectre_input(image, mask)
        tokens, _ = encode_spectre(
            encoder, prepared.data, expected_grid=prepared.grid_size
        )
        patient_embedding = tokens[0]
        patch_embeddings = tokens[1:]
        dimension = SPECTRE_EMBEDDING_DIMENSION
        valid_counts = np.full(
            patch_embeddings.shape[0], int(np.prod(SPECTRE_CROP_SIZE)), dtype=np.int64
        )
    elif encoder_name is ImageEncoderName.MERLIN:
        prepared = prepare_merlin_input(image, mask)
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
        "roi_target": case.roi_target,
        "native_roi_shape": native_shape,
        "resampled_shape": prepared.metadata.get("target_shape"),
        "patch_count": int(patch_embeddings.shape[0]),
        "embedding_dimension": dimension,
        "inference_seconds": round(inference_seconds, 3),
        "roi_ct_sha256": roi_sha256,
    }
    record = {
        **summary,
        "roi_pancreas_mask_sha256": mask_sha256,
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
    encoder_name: ImageEncoderName, artifacts: FoundationModelArtifacts
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
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
                "centering": "predicted pancreas bounding-box center",
                "margin_mm": 15.0,
                "padding": "-1000 HU to whole crop-grid multiples",
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
            "centering": "predicted pancreas bounding-box center",
            "padding": "-1000 HU before intensity scaling",
        },
        MERLIN_EMBEDDING_DIMENSION,
    )


def run_embedding(
    roi_run: Path,
    output_root: Path,
    model_cache: Path,
    *,
    encoder_name: ImageEncoderName = ImageEncoderName.SPECTRE,
    run_dir: Path | None = None,
    patients: set[str] | None = None,
    resume: bool = True,
    force: bool = False,
    local_model_only: bool = False,
) -> Path:
    cases = discover_roi_cases(roi_run, patients)
    if any(case.roi_target != "pancreas" for case in cases):
        raise ValueError("SPECTRE and Merlin require an ROI run with roi_target=pancreas")
    destination = run_dir or create_run_directory(output_root / encoder_name.value)
    destination.mkdir(parents=True, exist_ok=True)
    state_dir = destination / ".state"
    if force and state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(exist_ok=True)

    foundation_artifacts = acquire_foundation_model(
        encoder_name, model_cache, local_files_only=local_model_only
    )
    encoder, runtime = load_foundation_runtime(encoder_name, foundation_artifacts)
    model_fingerprint = ":".join(foundation_artifacts.hashes)
    model_manifest, feature_manifest, preprocessing_manifest, dimension = (
        _foundation_manifest(encoder_name, foundation_artifacts)
    )

    completed: list[CaseEncoding] = []
    rows: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    started_run = time.perf_counter()
    for case in cases:
        roi_sha256 = sha256_file(case.roi_ct_path)
        mask_sha256 = (
            sha256_file(case.roi_pancreas_mask_path)
            if case.roi_pancreas_mask_path is not None
            else None
        )
        state_json, state_npz = _state_paths(state_dir, case.patient_id)
        cached = (
            _load_cached_case(
                state_json,
                state_npz,
                roi_sha256=roi_sha256,
                mask_sha256=mask_sha256,
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
            continue
        try:
            encoded = _encode_foundation_case(encoder, case, encoder_name)
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
                    "roi_ct_sha256": roi_sha256,
                    "roi_pancreas_mask_sha256": mask_sha256,
                    "encoder": encoder_name.value,
                    "model_fingerprint": model_fingerprint,
                    "summary": encoded.summary,
                    "record": encoded.record,
                },
                state_json,
            )
        except Exception as error:
            failure = {
                "patient_id": case.patient_id,
                "status": "failed",
                "reason": f"{type(error).__name__}: {error}",
                "encoder": encoder_name.value,
                "roi_target": case.roi_target,
                "roi_ct_sha256": roi_sha256,
            }
            rows.append(failure)
            records.append(failure)
            failures.append(failure)

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
    write_summary(rows, destination / "embedding_summary.csv")
    source_manifest = roi_run / "run_manifest.json"
    status_counts = dict(Counter(row["status"] for row in rows))
    write_json(
        {
            "pipeline_version": __version__,
            "stage": f"{encoder_name.value}_pancreas_roi_embedding",
            "encoder": encoder_name.value,
            "status": "complete" if not failures else "completed_with_failures",
            "roi_run": str(roi_run.resolve()),
            "roi_run_manifest_sha256": sha256_file(source_manifest),
            "strict_pancreas_mask_required": True,
            "model": model_manifest,
            "feature_extraction": feature_manifest,
            "preprocessing": preprocessing_manifest,
            "runtime": runtime.to_dict(),
            "patient_count": len(completed),
            "patch_count": int(patch_embeddings.shape[0]),
            "status_counts": status_counts,
            "patients": records,
            "failures": failures,
            "elapsed_seconds": time.perf_counter() - started_run,
            "artifacts": {
                "patient_embeddings": "image_embeddings.npz",
                "patch_embeddings": "patch_embeddings.npz",
                "summary": "embedding_summary.csv",
            },
            "provisional_research_output": True,
        },
        destination / "run_manifest.json",
    )
    return destination
