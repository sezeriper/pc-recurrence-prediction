from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pc_recurrence.io import sha256_file, write_json, write_summary
from pc_recurrence.text_embedding import pipeline
from pc_recurrence.text_embedding.constants import EMBEDDING_DIMENSION, MODEL_REVISION
from pc_recurrence.text_embedding.model import TextModelArtifacts, TextRuntimeInfo

RAW_COLUMNS = ("patient_id", "ct_report_text", "ct_report_missing")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_clinical_run(tmp_path: Path, *, missing_train_report: bool = False) -> Path:
    run = tmp_path / "clinical_run"
    run.mkdir()
    train = run / "clinical_raw_train.csv"
    validation = run / "clinical_raw_validation.csv"
    write_summary(
        [
            {
                "patient_id": "Patient 1",
                "ct_report_text": "Birinci BT raporu.",
                "ct_report_missing": 0,
            },
            {
                "patient_id": "Patient 2",
                "ct_report_text": "" if missing_train_report else "İkinci BT raporu.",
                "ct_report_missing": int(missing_train_report),
            },
        ],
        train,
        RAW_COLUMNS,
    )
    write_summary(
        [
            {
                "patient_id": "Patient 3",
                "ct_report_text": "Üçüncü BT raporu.",
                "ct_report_missing": 0,
            }
        ],
        validation,
        RAW_COLUMNS,
    )
    write_json(
        {
            "stage": "clinical_feature_preprocessing",
            "status": "complete",
            "split": {
                "train_patient_ids": ["Patient 1", "Patient 2"],
                "validation_patient_ids": ["Patient 3"],
            },
            "artifacts": {
                "clinical_raw_train": {"path": train.name, "sha256": sha256_file(train)},
                "clinical_raw_validation": {
                    "path": validation.name,
                    "sha256": sha256_file(validation),
                },
            },
        },
        run / "run_manifest.json",
    )
    return run


def _mock_qwen_runtime(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    artifacts = TextModelArtifacts(
        model_directory=model_dir,
        revision=MODEL_REVISION,
        config_sha256="a" * 64,
        tokenizer_sha256="b" * 64,
        model_index_sha256="c" * 64,
    )
    runtime = TextRuntimeInfo(
        torch_version="test",
        transformers_version="test",
        device_type="cpu",
        device_name="test",
        total_device_bytes=None,
        free_device_bytes=None,
        load_seconds=0.1,
        model_dtype="float32",
    )
    batches: list[list[str]] = []
    monkeypatch.setattr(pipeline, "acquire_qwen_model", lambda *_args, **_kwargs: artifacts)
    monkeypatch.setattr(
        pipeline,
        "load_qwen_runtime",
        lambda _artifacts: (object(), object(), runtime),
    )
    monkeypatch.setattr(
        pipeline,
        "report_token_details",
        lambda _tokenizer, text, max_length: (len(text), len(text) > max_length),
    )

    def _encode(_tokenizer, _model, texts: list[str], *, max_length: int) -> np.ndarray:
        batches.append(texts)
        vectors = np.zeros((len(texts), EMBEDDING_DIMENSION), dtype=np.float32)
        for index in range(len(texts)):
            vectors[index, index] = 1.0
        return vectors

    monkeypatch.setattr(pipeline, "encode_reports", _encode)
    return batches


def test_text_embedding_writes_split_aligned_npz_and_audit_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clinical_run = _write_clinical_run(tmp_path)
    batches = _mock_qwen_runtime(tmp_path, monkeypatch)
    destination = pipeline.run_text_embedding(
        clinical_run,
        run_dir=tmp_path / "text_run",
        batch_size=2,
    )

    with np.load(destination / "text_embeddings_train.npz", allow_pickle=False) as train:
        assert train["patient_ids"].tolist() == ["Patient 1", "Patient 2"]
        assert train["embeddings"].dtype == np.float32
        assert train["embeddings"].shape == (2, EMBEDDING_DIMENSION)
        assert train["encoder"].item() == "qwen3-embedding-8b"
        assert train["split"].item() == "train"
    with np.load(destination / "text_embeddings_validation.npz", allow_pickle=False) as validation:
        assert validation["patient_ids"].tolist() == ["Patient 3"]
        assert validation["embeddings"].shape == (1, EMBEDDING_DIMENSION)
        assert validation["split"].item() == "validation"
    assert batches == [["Birinci BT raporu.", "İkinci BT raporu."], ["Üçüncü BT raporu."]]

    summary = _read_csv(destination / "embedding_summary.csv")
    assert [row["patient_id"] for row in summary] == ["Patient 1", "Patient 2", "Patient 3"]
    assert [row["split"] for row in summary] == ["train", "train", "validation"]
    assert all(row["status"] == "embedded" for row in summary)
    assert all("raporu" not in value.casefold() for row in summary for value in row.values())
    manifest = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "qwen3_ct_report_embedding"
    assert manifest["model"]["revision"] == MODEL_REVISION
    assert manifest["feature_extraction"]["pooling"] == "last-token"
    assert manifest["feature_extraction"]["l2_normalization"] is True
    for entry in manifest["artifacts"].values():
        assert entry["sha256"] == sha256_file(destination / entry["path"])


def test_missing_report_is_skipped_or_rejected_by_policy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clinical_run = _write_clinical_run(tmp_path, missing_train_report=True)
    batches = _mock_qwen_runtime(tmp_path, monkeypatch)
    destination = pipeline.run_text_embedding(clinical_run, run_dir=tmp_path / "text_run")

    summary = _read_csv(destination / "embedding_summary.csv")
    assert [row["status"] for row in summary] == ["embedded", "skipped", "embedded"]
    assert summary[1]["reason"] == "clinical raw table has no CT report text"
    with np.load(destination / "text_embeddings_train.npz", allow_pickle=False) as train:
        assert train["patient_ids"].tolist() == ["Patient 1"]
    assert batches == [["Birinci BT raporu."], ["Üçüncü BT raporu."]]

    with pytest.raises(ValueError, match="CT report text is unavailable"):
        pipeline.run_text_embedding(
            clinical_run,
            run_dir=tmp_path / "require_all",
            skip_unavailable=False,
        )


def test_cached_embeddings_are_reused_when_inputs_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clinical_run = _write_clinical_run(tmp_path)
    batches = _mock_qwen_runtime(tmp_path, monkeypatch)
    destination = tmp_path / "text_run"
    pipeline.run_text_embedding(clinical_run, run_dir=destination)
    pipeline.run_text_embedding(clinical_run, run_dir=destination)

    assert batches == [["Birinci BT raporu."], ["İkinci BT raporu."], ["Üçüncü BT raporu."]]


def test_rejects_invalid_max_length_before_model_acquisition(tmp_path: Path) -> None:
    clinical_run = _write_clinical_run(tmp_path)

    with pytest.raises(ValueError, match="max_length"):
        pipeline.run_text_embedding(
            clinical_run,
            run_dir=tmp_path / "text_run",
            max_length=0,
        )
