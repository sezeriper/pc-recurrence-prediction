from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from pydicom.uid import generate_uid
from test_image_dicom import _write_slice
from typer.testing import CliRunner

from pc_recurrence.image_data.cli import app
from pc_recurrence.image_data.constants import (
    DEFAULT_DICOM_ROOT,
    DEFAULT_INSPECTION_OUTPUT_ROOT,
)
from pc_recurrence.image_data.inspection import inspect_dataset
from pc_recurrence.image_data.preprocess import curate_dataset, write_curation_report
from pc_recurrence.image_data.workbook import EXPECTED_HEADERS, IMPORTANT_SLICES_HEADERS


def _legacy_row(patient_id: str, number: int, image_range: str) -> list[Any]:
    row: list[Any] = [None] * len(EXPECTED_HEADERS)
    row[0] = patient_id
    row[1] = number
    row[4] = "yok"
    row[15] = image_range
    return row


def _write_legacy_workbook(path: Path, rows: list[list[Any]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sayfa1"
    sheet.append(EXPECTED_HEADERS)
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def _write_important_slices_workbook(path: Path, case_id: str, image_range: str) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sayfa1"
    sheet.append(IMPORTANT_SLICES_HEADERS)
    sheet.append([case_id, 1, 60, 3, 1, 0.5, None, 4, 1000, 2, 100, 50, image_range, None])
    workbook.save(path)


def _write_series(root: Path, folder: str, count: int, *, prefix: str = "IM") -> tuple[str, str]:
    study_uid = generate_uid()
    series_uid = generate_uid()
    for index in range(1, count + 1):
        _write_slice(
            root / folder / f"{prefix}{index:06d}.dcm",
            study_uid=study_uid,
            series_uid=series_uid,
            z=float(index),
            stored_value=index,
            instance_number=index,
        )
    return study_uid, series_uid


def test_writable_defaults_are_outside_the_source_dataset() -> None:
    source = Path("../Radyoloji Data/10Patients_Important_Slices").resolve()
    assert DEFAULT_DICOM_ROOT.resolve().is_relative_to(Path.cwd().resolve())
    assert DEFAULT_INSPECTION_OUTPUT_ROOT.resolve().is_relative_to(Path.cwd().resolve())
    assert not DEFAULT_DICOM_ROOT.resolve().is_relative_to(source)
    assert not DEFAULT_INSPECTION_OUTPUT_ROOT.resolve().is_relative_to(source)


def test_direct_curation_applies_legacy_ranges_without_a_selection_file(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    study_uid, series_uid = _write_series(source, "PATIENT111", 4)
    workbook = tmp_path / "workbook.xlsx"
    _write_legacy_workbook(workbook, [_legacy_row("Patient 1", 111, "1-2")])

    output = tmp_path / "curated"
    report = curate_dataset(source, output, workbook)

    patient = report.patients[0]
    assert patient.status == "copied"
    assert patient.study_uid == study_uid
    assert patient.series_uid == series_uid
    assert patient.selected_file_count == 2
    assert sorted(path.name for path in (output / "PATIENT111").iterdir()) == [
        "IM000002.dcm",
        "IM000003.dcm",
    ]


def test_direct_curation_retains_all_professionally_selected_slices(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _write_series(source, "CASE_ABC", 3)
    workbook = tmp_path / "10Patients_Tabular.xlsx"
    _write_important_slices_workbook(workbook, "CASE_ABC", "420-572")

    report = curate_dataset(source, tmp_path / "curated", workbook)

    patient = report.patients[0]
    assert patient.status == "copied"
    assert patient.selected_file_count == 3
    assert patient.slices_are_preselected


def test_direct_curation_skips_missing_or_ambiguous_ct_folders(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _write_series(source, "PATIENT111", 3, prefix="one")
    _write_series(source, "PATIENT111", 3, prefix="two")
    workbook = tmp_path / "workbook.xlsx"
    _write_legacy_workbook(
        workbook,
        [_legacy_row("Patient 1", 111, "0-1"), _legacy_row("Patient 2", 222, "0-1")],
    )

    report = curate_dataset(source, tmp_path / "curated", workbook)

    assert [patient.status for patient in report.patients] == ["skipped", "skipped"]
    assert "expected one processable" in (report.patients[0].reason or "")
    assert report.patients[1].reason == "professionally curated CT folder is missing"


def test_curation_is_idempotent_and_records_direct_source_policy(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _write_series(source, "PATIENT111", 4)
    workbook = tmp_path / "workbook.xlsx"
    _write_legacy_workbook(workbook, [_legacy_row("Patient 1", 111, "0-2")])
    output = tmp_path / "curated"

    first = curate_dataset(source, output, workbook)
    second = curate_dataset(source, output, workbook)
    forced = curate_dataset(source, output, workbook, force=True)

    assert first.patients[0].copied_file_count == 3
    assert second.patients[0].unchanged_file_count == 3
    assert forced.patients[0].copied_file_count == 3
    _, manifest_path = write_curation_report(forced)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["policy"]["series_selection"] == (
        "one processable CT Series per professionally curated folder"
    )
    assert "selection_path" not in manifest


def test_subset_and_cli_preprocess_do_not_expose_review_commands(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _write_series(source, "PATIENT111", 3)
    _write_series(source, "PATIENT222", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_legacy_workbook(
        workbook,
        [_legacy_row("Patient 1", 111, "0-1"), _legacy_row("Patient 2", 222, "0-1")],
    )
    output = tmp_path / "curated"

    report = curate_dataset(source, output, workbook, patients={"PATIENT222"})
    assert [patient.patient_id for patient in report.patients] == ["Patient 2"]
    assert (output / "PATIENT222").is_dir()
    assert not (output / "PATIENT111").exists()

    runner = CliRunner()
    result = runner.invoke(
        app,
        [
            "preprocess",
            "--dicom-root",
            str(source),
            "--workbook",
            str(workbook),
            "--output-dir",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    help_result = runner.invoke(app, ["--help"])
    assert "inventory" not in help_result.output
    assert "review" not in help_result.output


def test_inspection_consumes_directly_curated_series(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    study_uid, series_uid = _write_series(source, "PATIENT111", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_legacy_workbook(workbook, [_legacy_row("Patient 1", 111, "0-2")])
    curated = tmp_path / "curated"
    write_curation_report(curate_dataset(source, curated, workbook))

    inspection = inspect_dataset(curated, workbook_path=workbook)[0]
    assert inspection.study_uid == study_uid
    assert inspection.series_uid == series_uid
    assert inspection.geometry_status == "eligible"
