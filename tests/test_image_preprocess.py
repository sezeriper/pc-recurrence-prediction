from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openpyxl import Workbook
from pydicom.uid import generate_uid
from test_image_dicom import _write_slice
from typer.testing import CliRunner

from pc_recurrence.image_roi.cli import app
from pc_recurrence.image_roi.pipeline import inspect_dataset
from pc_recurrence.image_roi.preprocess import curate_dataset, write_curation_report
from pc_recurrence.image_roi.workbook import EXPECTED_HEADERS


def _row(patient_id: str, hasta_no: int, image_range: str | None) -> list[Any]:
    return [
        patient_id,
        hasta_no,
        "whipple",
        "adenokarsinom",
        "yok",
        60,
        10,
        1,
        0.5,
        "yok",
        4,
        1500,
        "yok",
        None,
        50,
        image_range,
        "Kitle 2 cm olabilir.",
    ]


def _write_workbook(path: Path, rows: list[list[Any]]) -> None:
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Sayfa1"
    sheet.append(list(EXPECTED_HEADERS))
    for row in rows:
        sheet.append(row)
    workbook.save(path)


def _patient_folder(
    root: Path,
    folder: str,
    count: int,
    *,
    z_spacing: float = 0.5,
    gap_after: int | None = None,
) -> Path:
    patient = root / folder
    patient.mkdir(parents=True)
    uid = generate_uid()
    z = 0.0
    for index in range(1, count + 1):
        if gap_after is not None and index == gap_after + 1:
            z += 20.0
        _write_slice(
            patient / f"IM{index:06d}.dcm",
            series_uid=uid,
            z=z,
            stored_value=index,
            instance_number=index,
        )
        z += z_spacing
    return patient


def test_curate_copies_exact_table_range(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _patient_folder(source, "PATIENT1234567", 5)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 1234567, "2-4")])

    report = curate_dataset(source, tmp_path / "curated", workbook)

    assert len(report.patients) == 1
    patient = report.patients[0]
    assert patient.status == "copied"
    assert patient.geometry_status == "eligible"
    assert patient.selected_file_count == 3
    assert patient.copied_file_count == 3
    assert patient.unchanged_file_count == 0
    assert patient.selected_instance_numbers == [2, 3, 4]
    destination = tmp_path / "curated" / "PATIENT1234567"
    assert sorted(path.name for path in destination.iterdir()) == [
        "IM000002.dcm",
        "IM000003.dcm",
        "IM000004.dcm",
    ]
    assert len(list((source / "PATIENT1234567").iterdir())) == 5


def test_curate_is_idempotent_and_force_overwrites(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _patient_folder(source, "PATIENT1234567", 4)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 1234567, "1-3")])
    output = tmp_path / "curated"

    first = curate_dataset(source, output, workbook)
    assert first.patients[0].copied_file_count == 3

    second = curate_dataset(source, output, workbook)
    assert second.patients[0].copied_file_count == 0
    assert second.patients[0].unchanged_file_count == 3
    assert [item.action for item in second.patients[0].files] == ["unchanged"] * 3

    destination = output / "PATIENT1234567" / "IM000002.dcm"
    destination.write_bytes(b"corrupt")
    repaired = curate_dataset(source, output, workbook)
    assert repaired.patients[0].copied_file_count == 1
    assert repaired.patients[0].unchanged_file_count == 2
    assert [item.action for item in repaired.patients[0].files].count("overwritten") == 1

    forced = curate_dataset(source, output, workbook, force=True)
    assert forced.patients[0].copied_file_count == 3
    assert all(item.action == "overwritten" for item in forced.patients[0].files)


def test_curate_skips_missing_folder_and_missing_range(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _patient_folder(source, "PATIENT2000000", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(
        workbook,
        [
            _row("Patient 1", 1000000, "1-2"),
            _row("Patient 2", 2000000, None),
        ],
    )
    output = tmp_path / "curated"

    report = curate_dataset(source, output, workbook)

    assert [patient.status for patient in report.patients] == ["skipped", "skipped"]
    assert report.patients[0].reason == "no source DICOM folder"
    assert report.patients[1].reason == "no workbook image range"
    assert not any(output.iterdir())


def test_curate_records_invalid_range_and_gap_without_aborting(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _patient_folder(source, "PATIENT1111111", 3)
    _patient_folder(source, "PATIENT2222222", 4, gap_after=2)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(
        workbook,
        [
            _row("Patient 1", 1111111, "2-9"),
            _row("Patient 2", 2222222, "1-4"),
        ],
    )
    output = tmp_path / "curated"

    report = curate_dataset(source, output, workbook)

    invalid, gap = report.patients
    assert invalid.status == "skipped"
    assert "exceeds 3 series slices" in invalid.reason
    assert gap.status == "copied"
    assert gap.geometry_status == "eligible"
    assert "gap" in (gap.geometry_reason or "")
    assert gap.selected_file_count == 4
    assert len(list((output / "PATIENT2222222").iterdir())) == 4
    assert not (output / "PATIENT1111111").exists()


def test_curate_writes_manifest_and_summary(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _patient_folder(source, "PATIENT1234567", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 1234567, "1-2")])
    output = tmp_path / "curated"

    report = curate_dataset(source, output, workbook)
    summary, manifest = write_curation_report(report)

    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["stage"] == "dicom_slice_curation"
    assert data["status"] == "complete"
    assert data["totals"]["copied_patient_count"] == 1
    assert data["patients"][0]["files"][0]["action"] == "copied"
    assert data["workbook_sha256"]
    lines = summary.read_text(encoding="utf-8-sig").splitlines()
    assert lines[0].startswith("patient_id,")
    assert len(lines) == 2


def test_curate_patients_filter(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _patient_folder(source, "PATIENT1234567", 3)
    _patient_folder(source, "PATIENT7654321", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(
        workbook,
        [
            _row("Patient 1", 1234567, "1-2"),
            _row("Patient 2", 7654321, "1-2"),
        ],
    )
    output = tmp_path / "curated"

    report = curate_dataset(source, output, workbook, patients={"Patient 2"})

    assert [patient.patient_id for patient in report.patients] == ["Patient 2"]
    assert (output / "PATIENT7654321").is_dir()
    assert not (output / "PATIENT1234567").exists()


def test_inspect_dataset_uses_curated_folder_as_is(tmp_path: Path) -> None:
    source = tmp_path / "dicom_orig"
    _patient_folder(source, "PATIENT1234567", 5)
    _patient_folder(source, "PATIENT2222222", 4, gap_after=2)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(
        workbook,
        [
            _row("Patient 1", 1234567, "2-4"),
            _row("Patient 2", 2222222, "1-4"),
        ],
    )
    curated = tmp_path / "curated"
    report = curate_dataset(source, curated, workbook)
    write_curation_report(report)

    inspections = inspect_dataset(curated, workbook_path=workbook)

    by_id = {item.patient_id: item for item in inspections}
    assert by_id["Patient 1"].geometry_status == "eligible"
    assert by_id["Patient 1"].file_count == 3
    assert by_id["Patient 2"].geometry_status == "eligible"
    assert "gap" in (by_id["Patient 2"].reason or "")
    assert by_id["Patient 2"].file_count == 4


def test_preprocess_cli_curates_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _patient_folder(source, "PATIENT1234567", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 1234567, "1-2")])
    output = tmp_path / "curated"

    result = CliRunner().invoke(
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

    assert result.exit_code == 0
    assert "Patient 1: copied" in result.output
    assert (output / "PATIENT1234567").is_dir()
    assert (output / "curation_manifest.json").is_file()


def test_preprocess_cli_help() -> None:
    result = CliRunner().invoke(app, ["preprocess", "--help"])

    assert result.exit_code == 0
    assert "curated folder" in result.output
