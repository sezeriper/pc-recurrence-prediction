from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from openpyxl import Workbook
from typer.testing import CliRunner

from pc_recurrence.image_data.workbook import EXPECTED_HEADERS, load_image_workbook
from pc_recurrence.image_embedding.constants import (
    MERLIN_EMBEDDING_DIMENSION,
    SPECTRE_EMBEDDING_DIMENSION,
)
from pc_recurrence.io import write_json, write_npz
from pc_recurrence.recurrence_classifier.cli import app
from pc_recurrence.recurrence_classifier.model import RecurrenceHead
from pc_recurrence.recurrence_classifier.pipeline import (
    parse_recurrence_label,
    predict_recurrence,
    train_recurrence_models,
)


def _workbook(path: Path, labels: dict[str, object]) -> Path:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sayfa1"
    sheet.append(EXPECTED_HEADERS)
    for index, (patient_id, label) in enumerate(labels.items(), start=1):
        row: list[object | None] = [None] * len(EXPECTED_HEADERS)
        row[0] = patient_id
        row[1] = index
        row[4] = label
        row[15] = "1-2"
        sheet.append(row)
    workbook.save(path)
    return path


def _embedding_run(
    path: Path,
    encoder: str,
    dimension: int,
    patient_ids: list[str],
    labels: dict[str, object],
    *,
    curation_manifest_hash: str = "shared-curation-manifest",
    centering: str = "volume",
) -> Path:
    path.mkdir()
    targets = np.asarray(
        [
            0 if str(labels[patient_id]).strip().casefold() == "yok" else 1
            for patient_id in patient_ids
        ],
        dtype=np.float32,
    )
    embeddings = np.zeros((len(patient_ids), dimension), dtype=np.float32)
    embeddings[:, 0] = np.where(targets == 1, 3.0, -3.0)
    embeddings[:, 1] = np.arange(len(patient_ids), dtype=np.float32) / 100
    write_npz(
        path / "image_embeddings.npz",
        patient_ids=np.asarray(patient_ids, dtype=np.str_),
        embeddings=embeddings,
        patch_counts=np.ones(len(patient_ids), dtype=np.int32),
        encoder=np.asarray(encoder),
    )
    patients = [
        {
            "patient_id": patient_id,
            "status": "embedded",
            "encoder": encoder,
            "embedding_dimension": dimension,
            "ct_series_sha256": f"ct-series-{patient_id}",
            "preprocessing": {"centering": centering},
        }
        for patient_id in patient_ids
    ]
    write_json(
        {
            "stage": f"{encoder}_selected_ct_embedding",
            "status": "complete",
            "encoder": encoder,
            "patient_count": len(patient_ids),
            "feature_extraction": {"embedding_dimension": dimension},
            "source_curation_manifest_sha256": curation_manifest_hash,
            "patients": patients,
            "artifacts": {"patient_embeddings": "image_embeddings.npz"},
        },
        path / "run_manifest.json",
    )
    return path


def _fixture_runs(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, object]]:
    labels: dict[str, object] = {
        f"Patient {index}": datetime(2024, 1, index) if index % 2 else " YOK "
        for index in range(1, 9)
    }
    workbook = _workbook(tmp_path / "patients.xlsx", labels)
    ids = list(labels)
    merlin = _embedding_run(
        tmp_path / "merlin", "merlin", MERLIN_EMBEDDING_DIMENSION, ids, labels
    )
    spectre = _embedding_run(
        tmp_path / "spectre", "spectre", SPECTRE_EMBEDDING_DIMENSION, ids, labels
    )
    return workbook, merlin, spectre, labels


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_recurrence_labels_and_workbook_values(tmp_path: Path) -> None:
    workbook = _workbook(
        tmp_path / "labels.xlsx",
        {"Patient 1": " YoK ", "Patient 2": datetime(2024, 2, 3), "Patient 3": "date"},
    )
    rows = load_image_workbook(workbook)
    assert [row.recurrence_raw for row in rows] == ["YoK", datetime(2024, 2, 3), "date"]
    parsed = [
        parse_recurrence_label(row.recurrence_raw, patient_id=row.patient_id) for row in rows
    ]
    assert parsed == [
        0,
        1,
        1,
    ]
    with pytest.raises(ValueError, match="Patient 4.*blank recurrence label"):
        parse_recurrence_label("  ", patient_id="Patient 4")


def test_recurrence_head_is_one_biased_affine_layer() -> None:
    head = RecurrenceHead(7)
    assert list(head.modules()) == [head, head.linear]
    assert tuple(head.linear.weight.shape) == (1, 7)
    assert head.linear.bias is not None
    assert head(torch.zeros((3, 7))).shape == (3,)
    assert head(torch.zeros((1, 7))).shape == (1,)


def test_train_and_predict_cli_end_to_end(tmp_path: Path) -> None:
    workbook, merlin, spectre, labels = _fixture_runs(tmp_path)
    model_run = tmp_path / "trained"
    runner = CliRunner()
    train_result = runner.invoke(
        app,
        [
            "train",
            "--merlin-run",
            str(merlin),
            "--spectre-run",
            str(spectre),
            "--workbook",
            str(workbook),
            "--run-dir",
            str(model_run),
        ],
    )
    assert train_result.exit_code == 0, train_result.output
    assert train_result.output.strip() == str(model_run.resolve())

    metrics = json.loads((model_run / "metrics.json").read_text())
    assert set(metrics) == {"evaluation_kind", "threshold", "split", "encoders"}
    assert metrics["evaluation_kind"] == "single_holdout"
    assert set(metrics["encoders"]) == {"merlin", "spectre"}
    assert metrics["encoders"]["merlin"]["roc_auc"] == 1.0
    assert metrics["encoders"]["spectre"]["roc_auc"] == 1.0
    assert set(metrics["split"]["train_patient_ids"]).isdisjoint(
        metrics["split"]["test_patient_ids"]
    )
    evaluation = _read_csv(model_run / "evaluation_predictions.csv")
    holdout_count = len(metrics["split"]["test_patient_ids"])
    assert [row["encoder"] for row in evaluation] == ["merlin"] * holdout_count + [
        "spectre"
    ] * holdout_count
    cohort = _read_csv(model_run / "cohort_classifications.csv")
    assert all(row["fit_scope"] == "full_cohort" for row in cohort)
    assert [row["patient_id"] for row in cohort[:8]] == list(labels)
    replay_run = predict_recurrence(
        model_run, merlin, spectre, tmp_path, run_dir=tmp_path / "replayed"
    )
    replay = _read_csv(replay_run / "classification_results.csv")
    assert [
        (row["encoder"], row["patient_id"], row["probability"], row["prediction"])
        for row in replay
    ] == [
        (row["encoder"], row["patient_id"], row["probability"], row["prediction"])
        for row in cohort
    ]


    new_merlin_ids = ["Patient 8", "Patient 2", "Patient 5"]
    new_spectre_ids = ["Patient 3", "Patient 1"]
    new_merlin = _embedding_run(
        tmp_path / "new_merlin",
        "merlin",
        MERLIN_EMBEDDING_DIMENSION,
        new_merlin_ids,
        labels,
    )
    new_spectre = _embedding_run(
        tmp_path / "new_spectre",
        "spectre",
        SPECTRE_EMBEDDING_DIMENSION,
        new_spectre_ids,
        labels,
    )
    prediction_run = tmp_path / "predicted"
    predict_result = runner.invoke(
        app,
        [
            "predict",
            "--model-run",
            train_result.output.strip(),
            "--merlin-run",
            str(new_merlin),
            "--spectre-run",
            str(new_spectre),
            "--run-dir",
            str(prediction_run),
        ],
    )
    assert predict_result.exit_code == 0, predict_result.output
    results = _read_csv(prediction_run / "classification_results.csv")
    assert [(row["encoder"], row["patient_id"]) for row in results] == [
        *(('merlin', patient_id) for patient_id in new_merlin_ids),
        *(('spectre', patient_id) for patient_id in new_spectre_ids),
    ]


def test_current_one_positive_cohort_split_and_undefined_metrics(tmp_path: Path) -> None:
    labels: dict[str, object] = {
        f"Patient {index}": datetime(2024, 1, 9) if index == 9 else "yok"
        for index in range(2, 11)
    }
    workbook = _workbook(tmp_path / "patients.xlsx", labels)
    patient_ids = list(labels)
    merlin = _embedding_run(
        tmp_path / "merlin",
        "merlin",
        MERLIN_EMBEDDING_DIMENSION,
        patient_ids,
        labels,
    )
    spectre = _embedding_run(
        tmp_path / "spectre",
        "spectre",
        SPECTRE_EMBEDDING_DIMENSION,
        patient_ids,
        labels,
    )
    model_run = train_recurrence_models(
        merlin, spectre, workbook, tmp_path, run_dir=tmp_path / "trained"
    )
    metrics = json.loads((model_run / "metrics.json").read_text())
    split = metrics["split"]
    assert split["seed"] == 0
    assert split["test_patient_ids"] == ["Patient 6", "Patient 7"]
    assert "Patient 9" in split["train_patient_ids"]
    for encoder in ("merlin", "spectre"):
        block = metrics["encoders"][encoder]
        assert block["sensitivity"] is None
        assert block["sensitivity_reason"] == "holdout contains no positive targets"
        assert block["roc_auc"] is None
        assert block["roc_auc_reason"] == "holdout contains fewer than two target classes"
        if block["true_positive"] + block["false_positive"]:
            assert block["precision"] == 0
            assert block["f1"] == 0
        else:
            assert block["precision"] is None
            assert block["precision_reason"] == "holdout contains no predicted positives"
            assert block["f1"] is None
            assert (
                block["f1_reason"]
                == "holdout contains no positive targets and no positive predictions"
            )


def test_embedding_and_provenance_validation_precedes_output(tmp_path: Path) -> None:
    workbook, merlin, spectre, _ = _fixture_runs(tmp_path)
    manifest_path = spectre / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["patients"][0]["ct_series_sha256"] = "different"
    write_json(manifest, manifest_path)
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="mismatched ct_series_sha256"):
        train_recurrence_models(merlin, spectre, workbook, tmp_path, run_dir=destination)
    assert not destination.exists()

    with np.load(merlin / "image_embeddings.npz", allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["embeddings"] = arrays["embeddings"].astype(np.float64)
    write_npz(merlin / "image_embeddings.npz", **arrays)
    with pytest.raises(ValueError, match="embeddings must be float32"):
        train_recurrence_models(merlin, spectre, workbook, tmp_path, run_dir=destination)
    assert not destination.exists()


def test_model_hash_and_output_conflicts_are_rejected(tmp_path: Path) -> None:
    workbook, merlin, spectre, _ = _fixture_runs(tmp_path)
    model_run = train_recurrence_models(
        merlin, spectre, workbook, tmp_path, run_dir=tmp_path / "trained"
    )
    conflict = tmp_path / "conflict"
    conflict.mkdir()
    with pytest.raises(FileExistsError, match="already exists"):
        predict_recurrence(model_run, merlin, spectre, tmp_path, run_dir=conflict)
    model_path = model_run / "models" / "merlin.npz"
    model_path.write_bytes(model_path.read_bytes() + b"tampered")
    destination = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="SHA-256"):
        predict_recurrence(model_run, merlin, spectre, tmp_path, run_dir=destination)
    assert not destination.exists()
