from __future__ import annotations

from pathlib import Path

from openpyxl import Workbook

from pc_recurrence.clinical_data.pipeline import preprocess_clinical_features
from pc_recurrence.clinical_data.workbook import load_clinical_workbook
from pc_recurrence.image_data.workbook import IMPORTANT_SLICES_HEADERS, load_image_workbook
from pc_recurrence.recurrence_classifier.pipeline import parse_recurrence_label


def _write_workbook(path: Path) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sayfa1"
    sheet.append(IMPORTANT_SLICES_HEADERS)
    sheet.append(
        [
            "CASE_ABC",
            0,
            61,
            2.5,
            1.2,
            0.4,
            "yok",
            4.1,
            1300,
            6.5,
            120,
            50,
            "420-572",
            "Example CT report.",
        ]
    )
    sheet.append(
        [
            "CASE_DEF",
            1,
            72,
            7,
            2.1,
            0.7,
            None,
            3.8,
            800,
            None,
            None,
            45,
            "10-20",
            None,
        ]
    )
    workbook.save(path)


def test_important_slices_schema_maps_case_ids_and_preserves_supplied_slices(
    tmp_path: Path,
) -> None:
    workbook = tmp_path / "10Patients_Tabular.xlsx"
    _write_workbook(workbook)

    rows = load_image_workbook(workbook)

    assert [row.patient_id for row in rows] == ["CASE_ABC", "CASE_DEF"]
    assert [row.dicom_folder for row in rows] == ["CASE_ABC", "CASE_DEF"]
    assert [row.recurrence_raw for row in rows] == [0, 1]
    assert rows[0].image_range_raw == "420-572"
    assert rows[0].selection_range_raw is None
    assert rows[0].slices_are_preselected
    labels = [
        parse_recurrence_label(row.recurrence_raw, patient_id=row.patient_id) for row in rows
    ]
    assert labels == [0, 1]


def test_important_slices_clinical_schema_maps_available_columns(tmp_path: Path) -> None:
    workbook = tmp_path / "10Patients_Tabular.xlsx"
    _write_workbook(workbook)

    rows = load_clinical_workbook(workbook)

    assert rows[0].patient_id == "CASE_ABC"
    assert rows[0].recurrence_raw == 0
    assert rows[0].age == 61
    assert rows[0].image_range == "420-572"
    assert rows[0].ct_report == "Example CT report."
    assert rows[0].surgery is None
    assert rows[0].pathology is None

    output = preprocess_clinical_features(
        workbook,
        run_dir=tmp_path / "clinical_features",
        validation_fraction=0.5,
    )
    raw_rows = (output / "clinical_raw.csv").read_text(encoding="utf-8-sig").splitlines()
    assert any(line.startswith("CASE_ABC,") and ",0," in line for line in raw_rows)
    assert any(line.startswith("CASE_DEF,") and ",1," in line for line in raw_rows)
