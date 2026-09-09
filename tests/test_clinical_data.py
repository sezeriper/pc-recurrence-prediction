from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook
from typer.testing import CliRunner

from pc_recurrence.clinical_data.cli import app
from pc_recurrence.clinical_data.pipeline import preprocess_clinical_features
from pc_recurrence.image_data.workbook import EXPECTED_HEADERS
from pc_recurrence.io import sha256_file


def _row(
    patient_id: str,
    hospital_number: int,
    *,
    surgery: str,
    pathology: str,
    recurrence: Any,
    age: Any,
    ca_19_9: Any,
    crp: Any,
    cally: Any,
    symptom: Any,
    report: Any,
) -> list[Any]:
    return [
        patient_id,
        hospital_number,
        surgery,
        pathology,
        recurrence,
        age,
        ca_19_9,
        1.0,
        0.5,
        symptom,
        4.0,
        1500.0,
        crp,
        cally,
        50.0,
        "10-20",
        report,
    ]


def _write_workbook(path: Path, rows: list[list[Any]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sayfa1"
    sheet.append(list(EXPECTED_HEADERS))
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _fixture_workbook(path: Path) -> None:
    _write_workbook(
        path,
        [
            _row(
                "Patient 1",
                101,
                surgery="whipple",
                pathology="duktal adenokarsinom",
                recurrence="yok",
                age=10,
                ca_19_9="<2",
                crp="yok",
                cally=None,
                symptom="sarılık, kaşıntı",
                report="Birinci BT raporu.",
            ),
            _row(
                "Patient 2",
                102,
                surgery="distal pank",
                pathology="duktal adenokarsinom",
                recurrence="2024-01-01",
                age=20,
                ca_19_9=4,
                crp=10,
                cally=100,
                symptom="yok",
                report="İkinci BT raporu.",
            ),
            _row(
                "Patient 3",
                103,
                surgery="total pankreatektomi",
                pathology="başka patoloji",
                recurrence="yok",
                age=None,
                ca_19_9=None,
                crp=20,
                cally=200,
                symptom=None,
                report=None,
            ),
        ],
    )


def test_preprocess_writes_normalized_one_hot_table_and_manifest(tmp_path: Path) -> None:
    workbook = tmp_path / "patients.xlsx"
    _fixture_workbook(workbook)
    destination = preprocess_clinical_features(
        workbook,
        run_dir=tmp_path / "clinical_run",
        validation_fraction=1 / 3,
    )

    train_rows = _read_csv(destination / "clinical_features_train.csv")
    validation_rows = _read_csv(destination / "clinical_features_validation.csv")
    raw_train_rows = _read_csv(destination / "clinical_raw_train.csv")
    raw_validation_rows = _read_csv(destination / "clinical_raw_validation.csv")
    raw_audit_rows = _read_csv(destination / "clinical_raw.csv")
    assert [row["patient_id"] for row in train_rows] == ["Patient 1", "Patient 2"]
    assert [row["patient_id"] for row in validation_rows] == ["Patient 3"]
    assert set(row["patient_id"] for row in train_rows).isdisjoint(
        row["patient_id"] for row in validation_rows
    )
    assert [row["target_recurrence"] for row in train_rows] == ["0", "1"]
    assert validation_rows[0]["target_recurrence"] == "0"
    assert train_rows[0]["ct_report_text"] == "Birinci BT raporu."
    assert validation_rows[0]["ct_report_missing"] == "1"
    assert float(train_rows[0]["age_zscore"]) == pytest.approx(-1.0)
    assert float(train_rows[1]["age_zscore"]) == pytest.approx(1.0)
    assert float(validation_rows[0]["age_zscore"]) == pytest.approx(0.0)
    assert validation_rows[0]["age_missing"] == "1"
    assert train_rows[0]["ca_19_9_below_detection_limit"] == "1"
    assert train_rows[0]["crp_missing"] == "1"
    assert float(train_rows[0]["crp_zscore"]) == pytest.approx(0.0)
    assert train_rows[0]["symptom__sarilik"] == "1"
    assert train_rows[0]["symptom__kasinti"] == "1"
    assert train_rows[1]["symptom__yok"] == "1"
    assert validation_rows[0]["symptom__missing"] == "1"
    assert validation_rows[0]["surgery__unknown"] == "1"
    assert validation_rows[0]["pathology__unknown"] == "1"
    assert raw_train_rows[0]["ca_19_9"] == "<2"
    assert raw_train_rows[0]["crp"] == "yok"
    assert raw_validation_rows[0]["ct_report_text"] == ""
    assert {row["split"] for row in raw_audit_rows} == {"train", "validation"}
    assert {row["patient_id"] for row in raw_audit_rows} == {
        "Patient 1",
        "Patient 2",
        "Patient 3",
    }
    assert "hospital_number" not in train_rows[0]
    assert "image_range" not in train_rows[0]

    parameters = json.loads(
        (destination / "preprocessing_parameters.json").read_text(encoding="utf-8")
    )
    assert parameters["text_embedding"]["status"] == "pending"
    assert parameters["numeric_fields"]["age"]["mean"] == pytest.approx(15.0)
    assert parameters["fit_patient_ids"] == ["Patient 1", "Patient 2"]
    manifest = json.loads((destination / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == "clinical_feature_preprocessing"
    assert manifest["patient_count"] == 3
    assert manifest["train_patient_count"] == 2
    assert manifest["validation_patient_count"] == 1
    assert manifest["artifacts"]["clinical_features_train"]["sha256"] == sha256_file(
        destination / "clinical_features_train.csv"
    )


def test_explicit_fit_subset_maps_unseen_categories_to_unknown(tmp_path: Path) -> None:
    workbook = tmp_path / "patients.xlsx"
    _fixture_workbook(workbook)
    destination = preprocess_clinical_features(
        workbook,
        run_dir=tmp_path / "clinical_run",
        validation_fraction=1 / 3,
        fit_patients={"Patient 2"},
    )

    train_rows = _read_csv(destination / "clinical_features_train.csv")
    validation_rows = _read_csv(destination / "clinical_features_validation.csv")
    patient_one = train_rows[0]
    patient_three = validation_rows[0]
    assert patient_one["surgery__unknown"] == "1"
    assert patient_three["surgery__unknown"] == "1"
    assert patient_three["pathology__unknown"] == "1"
    assert "surgery__total_pankreatektomi" not in patient_three
    assert float(patient_one["age_zscore"]) == pytest.approx(-10.0)
    assert float(patient_three["age_zscore"]) == pytest.approx(0.0)
    parameters = json.loads(
        (destination / "preprocessing_parameters.json").read_text(encoding="utf-8")
    )
    assert parameters["fit_scope"] == "explicit_training_subset"
    assert parameters["fit_patient_ids"] == ["Patient 2"]


def test_invalid_numeric_value_fails_with_row_and_field(tmp_path: Path) -> None:
    workbook = tmp_path / "patients.xlsx"
    _write_workbook(
        workbook,
        [
            _row(
                "Patient 1",
                101,
                surgery="whipple",
                pathology="adenokarsinom",
                recurrence="yok",
                age="old",
                ca_19_9=2,
                crp=1,
                cally=2,
                symptom="yok",
                report="Rapor.",
            )
        ],
    )

    with pytest.raises(ValueError, match=r"Row 2.*'yaş'.*'old'"):
        preprocess_clinical_features(workbook, run_dir=tmp_path / "clinical_run")


def test_cli_generates_requested_run_directory(tmp_path: Path) -> None:
    workbook = tmp_path / "patients.xlsx"
    _fixture_workbook(workbook)
    destination = tmp_path / "clinical_run"

    result = CliRunner().invoke(
        app,
        [
            "preprocess",
            "--workbook",
            str(workbook),
            "--run-dir",
            str(destination),
            "--validation-fraction",
            "0.3333333333",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (destination / "clinical_features_train.csv").is_file()
    assert (destination / "clinical_features_validation.csv").is_file()
    assert (destination / "clinical_raw_train.csv").is_file()
    assert (destination / "clinical_raw_validation.csv").is_file()
    assert (destination / "clinical_raw.csv").is_file()
    assert str(destination.resolve()) in result.output
