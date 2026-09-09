from __future__ import annotations

import csv
import hashlib
import shutil
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from pc_recurrence import __version__
from pc_recurrence.io import (
    create_run_directory,
    read_json,
    sha256_file,
    write_json,
    write_npz,
    write_summary,
)

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_CACHE,
    DEFAULT_OUTPUT_ROOT,
    EMBEDDING_DIMENSION,
    MODEL_CONTEXT_LENGTH,
    MODEL_LABEL,
)
from .model import (
    TextModelArtifacts,
    acquire_qwen_model,
    encode_reports,
    load_qwen_runtime,
    report_token_details,
)

ProgressReporter = Callable[[str], None]

SUMMARY_COLUMNS = (
    "patient_id",
    "split",
    "status",
    "reason",
    "report_sha256",
    "report_character_count",
    "report_token_count",
    "truncated",
    "embedding_dimension",
    "inference_seconds",
)


@dataclass(frozen=True)
class CtReportCase:
    patient_id: str
    split: str
    report_text: str | None


@dataclass(frozen=True)
class EncodedReport:
    case: CtReportCase
    embedding: np.ndarray
    summary: dict[str, Any]
    record: dict[str, Any]


def _report(progress: ProgressReporter | None, message: str) -> None:
    if progress is not None:
        progress(f"[pc-text-embed] {message}")


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


def _artifact_path(clinical_run: Path, manifest: dict[str, Any], key: str) -> Path:
    artifact = manifest.get("artifacts", {}).get(key)
    if not isinstance(artifact, dict):
        raise ValueError(f"clinical preprocessing manifest has no {key!r} artifact")
    name = artifact.get("path")
    expected_hash = artifact.get("sha256")
    if not isinstance(name, str) or not isinstance(expected_hash, str):
        raise ValueError(f"clinical preprocessing artifact {key!r} is invalid")
    path = (clinical_run / name).resolve()
    if not path.is_relative_to(clinical_run.resolve()):
        raise ValueError(f"clinical preprocessing artifact {key!r} escapes its run directory")
    _require_file(path, f"clinical preprocessing artifact {key}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise ValueError(f"clinical preprocessing artifact {key!r} SHA-256 does not match")
    return path


def load_ct_report_cases(clinical_run: Path) -> tuple[list[CtReportCase], dict[str, Any]]:
    """Load report documents from a completed, hash-verified clinical preprocessing run."""
    clinical_run = Path(clinical_run)
    manifest_path = clinical_run / "run_manifest.json"
    _require_file(manifest_path, "clinical preprocessing manifest")
    manifest = read_json(manifest_path)
    if manifest.get("stage") != "clinical_feature_preprocessing":
        raise ValueError("clinical input manifest is not a clinical_feature_preprocessing run")
    if manifest.get("status") != "complete":
        raise ValueError("clinical input manifest is not complete")
    cases: list[CtReportCase] = []
    seen_ids: set[str] = set()
    for split in ("train", "validation"):
        raw_path = _artifact_path(clinical_run, manifest, f"clinical_raw_{split}")
        with raw_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            expected = {"patient_id", "ct_report_text", "ct_report_missing"}
            if reader.fieldnames is None or not expected.issubset(reader.fieldnames):
                raise ValueError(f"{raw_path} lacks required CT-report columns")
            for row_number, row in enumerate(reader, start=2):
                patient_id = (row.get("patient_id") or "").strip()
                if not patient_id:
                    raise ValueError(f"{raw_path} row {row_number} has no patient_id")
                if patient_id in seen_ids:
                    raise ValueError(f"clinical raw tables duplicate patient {patient_id!r}")
                seen_ids.add(patient_id)
                report = (row.get("ct_report_text") or "").strip()
                missing_flag = row.get("ct_report_missing")
                if missing_flag not in {"0", "1"}:
                    raise ValueError(
                        f"{raw_path} row {row_number} has invalid ct_report_missing "
                        f"{missing_flag!r}"
                    )
                if bool(report) == (missing_flag == "1"):
                    raise ValueError(
                        f"{raw_path} row {row_number} has inconsistent CT-report text "
                        "and missing flag"
                    )
                cases.append(CtReportCase(patient_id, split, report or None))
    if not cases:
        raise ValueError("clinical preprocessing tables contain no patient rows")
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise ValueError("clinical preprocessing manifest has no split provenance")
    expected_ids = {
        "train": split.get("train_patient_ids"),
        "validation": split.get("validation_patient_ids"),
    }
    for name, ids in expected_ids.items():
        actual = [case.patient_id for case in cases if case.split == name]
        if not isinstance(ids, list) or actual != ids:
            raise ValueError(f"clinical raw {name} patient order does not match its manifest")
    return cases, manifest


def _report_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _state_paths(state_dir: Path, patient_id: str) -> tuple[Path, Path]:
    safe_id = "".join(character if character.isalnum() else "_" for character in patient_id)
    return state_dir / f"{safe_id}.json", state_dir / f"{safe_id}.npz"


def _load_cached_case(
    state_json: Path,
    state_npz: Path,
    *,
    case: CtReportCase,
    report_sha256: str,
    model_fingerprint: str,
    max_length: int,
) -> EncodedReport | None:
    if not state_json.is_file() or not state_npz.is_file():
        return None
    try:
        metadata = read_json(state_json)
        if (
            metadata.get("patient_id") != case.patient_id
            or metadata.get("split") != case.split
            or metadata.get("report_sha256") != report_sha256
            or metadata.get("model_fingerprint") != model_fingerprint
            or metadata.get("max_length") != max_length
        ):
            return None
        with np.load(state_npz, allow_pickle=False) as archive:
            if set(archive.files) != {"embedding"}:
                return None
            embedding = archive["embedding"]
        if (
            embedding.dtype != np.float32
            or embedding.shape != (EMBEDDING_DIMENSION,)
            or not np.isfinite(embedding).all()
        ):
            return None
        return EncodedReport(
            case=case,
            embedding=embedding,
            summary=metadata["summary"],
            record=metadata["record"],
        )
    except (OSError, ValueError, KeyError):
        return None


def _missing_report(case: CtReportCase) -> tuple[dict[str, Any], dict[str, Any]]:
    summary = {
        "patient_id": case.patient_id,
        "split": case.split,
        "status": "skipped",
        "reason": "clinical raw table has no CT report text",
        "report_sha256": None,
        "report_character_count": 0,
        "report_token_count": None,
        "truncated": None,
        "embedding_dimension": None,
        "inference_seconds": None,
    }
    return summary, summary.copy()


def _destination(output_root: Path, run_dir: Path | None, resume: bool) -> Path:
    if run_dir is None:
        return create_run_directory(output_root / MODEL_LABEL)
    if run_dir.exists() and not resume:
        raise FileExistsError(f"Run directory already exists: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _write_split_embeddings(
    destination: Path,
    split: str,
    encoded: list[EncodedReport],
) -> Path:
    patient_ids = np.asarray([item.case.patient_id for item in encoded], dtype=np.str_)
    embeddings = (
        np.stack([item.embedding for item in encoded]).astype(np.float32)
        if encoded
        else np.empty((0, EMBEDDING_DIMENSION), dtype=np.float32)
    )
    return write_npz(
        destination / f"text_embeddings_{split}.npz",
        patient_ids=patient_ids,
        embeddings=embeddings,
        encoder=np.asarray(MODEL_LABEL),
        split=np.asarray(split),
    )


def run_text_embedding(
    clinical_run: Path,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    model_cache: Path = DEFAULT_MODEL_CACHE,
    *,
    run_dir: Path | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = DEFAULT_MAX_LENGTH,
    resume: bool = True,
    force: bool = False,
    skip_unavailable: bool = True,
    local_model_only: bool = False,
    progress: ProgressReporter | None = None,
) -> Path:
    """Embed split clinical CT reports with the pinned Qwen3-Embedding-8B model."""
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if not 1 <= max_length <= MODEL_CONTEXT_LENGTH:
        raise ValueError(
            f"max_length must be between 1 and {MODEL_CONTEXT_LENGTH}; received {max_length}"
        )
    _report(progress, f"Validating clinical preprocessing run {clinical_run}.")
    cases, clinical_manifest = load_ct_report_cases(Path(clinical_run))
    unavailable = [case for case in cases if case.report_text is None]
    if unavailable and not skip_unavailable:
        names = ", ".join(case.patient_id for case in unavailable)
        raise ValueError(f"CT report text is unavailable for: {names}")
    destination = _destination(Path(output_root), run_dir, resume)
    state_dir = destination / ".state"
    if force and state_dir.exists():
        shutil.rmtree(state_dir)
    state_dir.mkdir(exist_ok=True)

    summaries_by_id: dict[str, dict[str, Any]] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    for case in unavailable:
        summary, record = _missing_report(case)
        summaries_by_id[case.patient_id] = summary
        records_by_id[case.patient_id] = record
        _report(progress, f"Skipping {case.patient_id}: no CT report text.")
    available = [case for case in cases if case.report_text is not None]
    completed_by_id: dict[str, EncodedReport] = {}

    artifacts: TextModelArtifacts | None = None
    runtime: dict[str, Any] | None = None
    if available:
        _report(progress, "Acquiring the pinned Qwen3-Embedding-8B snapshot.")
        artifacts = acquire_qwen_model(Path(model_cache), local_files_only=local_model_only)
        model_fingerprint = artifacts.fingerprint
        pending: list[tuple[CtReportCase, str]] = []
        for case in available:
            assert case.report_text is not None
            report_hash = _report_sha256(case.report_text)
            state_json, state_npz = _state_paths(state_dir, case.patient_id)
            cached = (
                _load_cached_case(
                    state_json,
                    state_npz,
                    case=case,
                    report_sha256=report_hash,
                    model_fingerprint=model_fingerprint,
                    max_length=max_length,
                )
                if resume and not force
                else None
            )
            if cached is not None:
                completed_by_id[case.patient_id] = cached
                _report(progress, f"{case.patient_id}: reused resumable embedding cache.")
            else:
                pending.append((case, report_hash))
        if pending:
            _report(progress, "Model snapshot verified; loading Qwen runtime.")
            tokenizer, model, runtime_info = load_qwen_runtime(artifacts)
            runtime = runtime_info.to_dict()
            for start in range(0, len(pending), batch_size):
                batch = pending[start : start + batch_size]
                batch_cases = [case for case, _ in batch]
                texts = [case.report_text for case in batch_cases]
                assert all(text is not None for text in texts)
                token_details = [
                    report_token_details(tokenizer, str(text), max_length) for text in texts
                ]
                _report(
                    progress,
                    f"Embedding reports [{start + 1}-{start + len(batch)}/{len(pending)}].",
                )
                started = time.perf_counter()
                vectors = encode_reports(
                    tokenizer,
                    model,
                    [str(text) for text in texts],
                    max_length=max_length,
                )
                elapsed = time.perf_counter() - started
                for (case, report_hash), vector, (token_count, truncated) in zip(
                    batch, vectors, token_details, strict=True
                ):
                    assert case.report_text is not None
                    summary = {
                        "patient_id": case.patient_id,
                        "split": case.split,
                        "status": "embedded",
                        "reason": None,
                        "report_sha256": report_hash,
                        "report_character_count": len(case.report_text),
                        "report_token_count": token_count,
                        "truncated": truncated,
                        "embedding_dimension": EMBEDDING_DIMENSION,
                        "inference_seconds": round(elapsed / len(batch), 3),
                    }
                    record = {
                        **summary,
                        "model_fingerprint": model_fingerprint,
                        "max_length": max_length,
                    }
                    encoded = EncodedReport(
                        case=case,
                        embedding=np.asarray(vector, dtype=np.float32),
                        summary=summary,
                        record=record,
                    )
                    completed_by_id[case.patient_id] = encoded
                    state_json, state_npz = _state_paths(state_dir, case.patient_id)
                    write_npz(state_npz, embedding=encoded.embedding)
                    write_json(
                        {
                            "patient_id": case.patient_id,
                            "split": case.split,
                            "report_sha256": report_hash,
                            "model_fingerprint": model_fingerprint,
                            "max_length": max_length,
                            "summary": summary,
                            "record": record,
                        },
                        state_json,
                    )
        if runtime is None:
            runtime = {"reused_from_state": True}

    completed = [completed_by_id[case.patient_id] for case in available]
    for item in completed:
        summaries_by_id[item.case.patient_id] = item.summary
        records_by_id[item.case.patient_id] = item.record
    rows = [summaries_by_id[case.patient_id] for case in cases]
    records = [records_by_id[case.patient_id] for case in cases]
    train_encoded = [item for item in completed if item.case.split == "train"]
    validation_encoded = [item for item in completed if item.case.split == "validation"]
    train_path = _write_split_embeddings(destination, "train", train_encoded)
    validation_path = _write_split_embeddings(destination, "validation", validation_encoded)
    summary_path = write_summary(rows, destination / "embedding_summary.csv", SUMMARY_COLUMNS)
    status_counts = dict(Counter(row["status"] for row in rows))

    def _artifact(path: Path) -> dict[str, str]:
        return {"path": path.name, "sha256": sha256_file(path)}

    clinical_run = Path(clinical_run)
    clinical_manifest_path = clinical_run / "run_manifest.json"
    write_json(
        {
            "pipeline_version": __version__,
            "stage": "qwen3_ct_report_embedding",
            "status": "complete" if not unavailable else "completed_with_skips",
            "clinical_run": str(clinical_run.resolve()),
            "clinical_run_manifest_sha256": sha256_file(clinical_manifest_path),
            "clinical_split": clinical_manifest["split"],
            "model": artifacts.to_dict() if artifacts is not None else None,
            "runtime": runtime,
            "feature_extraction": {
                "input_column": "ct_report_text",
                "input_kind": "document",
                "instruction": None,
                "pooling": "last-token",
                "l2_normalization": True,
                "embedding_dimension": EMBEDDING_DIMENSION,
                "max_length": max_length,
                "batch_size": batch_size,
            },
            "skip_unavailable": skip_unavailable,
            "patient_count": len(cases),
            "embedded_patient_count": len(completed),
            "status_counts": status_counts,
            "patients": records,
            "artifacts": {
                "train_embeddings": _artifact(train_path),
                "validation_embeddings": _artifact(validation_path),
                "summary": _artifact(summary_path),
            },
            "provisional_research_output": True,
        },
        destination / "run_manifest.json",
    )
    _report(
        progress,
        f"Finished: {len(completed)} embedded and {len(unavailable)} skipped. "
        f"Artifacts: {destination}",
    )
    return destination
