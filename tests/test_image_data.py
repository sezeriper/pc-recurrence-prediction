from __future__ import annotations

import csv
import inspect as python_inspect
import json
from pathlib import Path
from typing import Any

import pytest
from openpyxl import Workbook
from pydicom.uid import SecondaryCaptureImageStorage, generate_uid
from test_image_dicom import DEFAULT_STUDY_UID, _write_slice
from typer.testing import CliRunner

from pc_recurrence.image_data import cli as image_cli
from pc_recurrence.image_data.cli import app
from pc_recurrence.image_data.constants import (
    DEFAULT_DICOM_ROOT,
    DEFAULT_INSPECTION_OUTPUT_ROOT,
    DEFAULT_SCAN_REVIEW_ROOT,
    DEFAULT_SCAN_SELECTION,
)
from pc_recurrence.image_data.dicom import SeriesKey
from pc_recurrence.image_data.inspection import inspect_dataset
from pc_recurrence.image_data.preprocess import (
    UNAVAILABLE_SKIP_REASON,
    curate_dataset,
    write_curation_report,
)
from pc_recurrence.image_data.scan_selection import (
    SCAN_SELECTION_COLUMNS,
    write_scan_inventory,
)
from pc_recurrence.image_data.workbook import EXPECTED_HEADERS


def test_writable_defaults_are_outside_dataset() -> None:
    assert Path("outputs/ct_series_review") == DEFAULT_SCAN_REVIEW_ROOT
    assert Path("outputs/ct_series_review/scan_selection.csv") == DEFAULT_SCAN_SELECTION
    assert Path("outputs/dicom_selected") == DEFAULT_DICOM_ROOT
    dataset_root = Path("dataset").resolve()
    writable_defaults = (
        DEFAULT_SCAN_REVIEW_ROOT,
        DEFAULT_SCAN_SELECTION,
        DEFAULT_DICOM_ROOT,
        DEFAULT_INSPECTION_OUTPUT_ROOT,
    )
    assert all(not path.resolve().is_relative_to(dataset_root) for path in writable_defaults)


def test_cli_writable_defaults_resolve_outside_dataset() -> None:
    command_defaults = {
        "inventory output": python_inspect.signature(image_cli.inventory)
        .parameters["output_dir"]
        .default,
        "review selection": python_inspect.signature(image_cli.review)
        .parameters["selection"]
        .default,
        "preprocess output": python_inspect.signature(image_cli.preprocess)
        .parameters["output_dir"]
        .default,
        "inspection output": python_inspect.signature(image_cli.inspect)
        .parameters["output_root"]
        .default,
    }
    dataset_root = Path("dataset").resolve()
    assert command_defaults["inventory output"] == Path("outputs/ct_series_review")
    assert command_defaults["review selection"] == Path(
        "outputs/ct_series_review/scan_selection.csv"
    )
    assert command_defaults["preprocess output"] == Path("outputs/dicom_selected")
    assert all(
        not default.resolve().is_relative_to(dataset_root) for default in command_defaults.values()
    )


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


def _write_series(
    root: Path,
    folder: str,
    count: int,
    *,
    series_uid: str | None = None,
    study_uid: str = DEFAULT_STUDY_UID,
    prefix: str = "IM",
    sop_class: str | None = None,
    gap_after: int | None = None,
    series_number: int | None = None,
) -> SeriesKey:
    patient = root / folder
    patient.mkdir(parents=True, exist_ok=True)
    series_uid = series_uid or generate_uid()
    z = 0.0
    for index in range(1, count + 1):
        if gap_after is not None and index == gap_after + 1:
            z += 20.0
        metadata = {"SeriesNumber": series_number} if series_number is not None else {}
        _write_slice(
            patient / f"{prefix}{index:06d}.dcm",
            series_uid=series_uid,
            study_uid=study_uid,
            z=z,
            stored_value=index,
            instance_number=index,
            sop_class=sop_class or "1.2.840.10008.5.1.4.1.1.2",
            metadata=metadata,
        )
        z += 0.5
    return SeriesKey(study_uid, series_uid)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SCAN_SELECTION_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _select(path: Path, patient_id: str, series_uid: str) -> None:
    rows = _read_csv(path)
    for row in rows:
        if row["patient_id"] == patient_id and row["series_instance_uid"] == series_uid:
            row["selected"] = "yes"
    _write_csv(path, rows)


def _inventory(
    source: Path,
    workbook: Path,
    review: Path,
    *,
    patients: set[str] | None = None,
    force: bool = False,
) -> Path:
    return write_scan_inventory(
        source, workbook, review, patients=patients, force=force
    ).selection_path


def test_inventory_writes_blank_choices_previews_and_classifications(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    ready = _write_series(source, "PATIENT111", 4, series_number=1)
    unsupported = _write_series(
        source,
        "PATIENT111",
        2,
        prefix="dose",
        sop_class=str(SecondaryCaptureImageStorage),
        series_number=2,
    )
    empty = source / "PATIENT222"
    empty.mkdir(parents=True)
    (empty / "zero.jpg").write_bytes(b"")
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(
        workbook,
        [_row("Patient 1", 111, "0-2"), _row("Patient 2", 222, "0-1")],
    )
    review = tmp_path / "review"

    report = write_scan_inventory(source, workbook, review)
    rows = _read_csv(report.selection_path)

    assert tuple(rows[0]) == SCAN_SELECTION_COLUMNS
    assert all(row["selected"] == "" for row in rows)
    by_uid = {row["series_instance_uid"]: row for row in rows if row["candidate_id"]}
    assert by_uid[ready.series_uid]["status"] == "ready"
    assert by_uid[ready.series_uid]["preview_status"] == "ready"
    assert (review / by_uid[ready.series_uid]["preview_path"]).is_file()
    assert by_uid[unsupported.series_uid]["status"] == "not_selectable"
    assert "unsupported SOPClassUID" in by_uid[unsupported.series_uid]["reason"]
    no_series = next(row for row in rows if row["patient_id"] == "Patient 2")
    assert no_series["status"] == "no_series"
    assert "ignored 1 files" in no_series["reason"]
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    assert manifest["stage"] == "ct_series_inventory"
    assert manifest["patient_count"] == 2
    assert manifest["candidate_count"] == 2


def test_inventory_refuses_overwrite_before_artifact_changes(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    _write_series(source, "PATIENT111", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 111, "0-1")])
    review = tmp_path / "review"
    selection = _inventory(source, workbook, review)
    before = selection.read_bytes()
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_scan_inventory(source, workbook, review)
    assert selection.read_bytes() == before


@pytest.mark.parametrize("mode", ["blank", "multiple", "invalid"])
def test_selection_preflight_fails_before_output_mutation(tmp_path: Path, mode: str) -> None:
    source = tmp_path / "dicom"
    first = _write_series(source, "PATIENT111", 3, prefix="a", series_number=1)
    second = _write_series(source, "PATIENT111", 3, prefix="b", series_number=2)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 111, "0-1")])
    selection = _inventory(source, workbook, tmp_path / "review")
    rows = _read_csv(selection)
    if mode == "multiple":
        for row in rows:
            row["selected"] = "yes"
    elif mode == "invalid":
        rows[0]["selected"] = "true"
    _write_csv(selection, rows)
    output = tmp_path / "curated"

    with pytest.raises(ValueError):
        curate_dataset(source, output, workbook, selection)
    assert not output.exists()
    assert first != second


def test_curation_can_skip_only_patients_without_ready_series(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    (source / "PATIENT111").mkdir(parents=True)
    selected = _write_series(source, "PATIENT222", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(
        workbook,
        [_row("Patient 1", 111, "0-1"), _row("Patient 2", 222, "0-1")],
    )
    selection = _inventory(source, workbook, tmp_path / "review")
    _select(selection, "Patient 2", selected.series_uid)

    output = tmp_path / "curated"
    report = curate_dataset(source, output, workbook, selection)

    assert [patient.status for patient in report.patients] == ["skipped", "copied"]
    assert report.patients[0].reason == UNAVAILABLE_SKIP_REASON
    assert (output / "PATIENT222").is_dir()
    assert not (output / "PATIENT111").exists()

    _, manifest_path = write_curation_report(report)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "complete_with_skips"
    assert manifest["policy"]["allow_unselected_without_ready_series"] is True


def test_exact_choice_applies_range_and_replaces_prior_series(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    first = _write_series(source, "PATIENT111", 5, prefix="first", series_number=1)
    second = _write_series(source, "PATIENT111", 4, prefix="second", series_number=2)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 111, "1-2")])
    review = tmp_path / "review"
    selection = _inventory(source, workbook, review)
    _select(selection, "Patient 1", first.series_uid)
    output = tmp_path / "curated"

    first_report = curate_dataset(source, output, workbook, selection)
    assert first_report.patients[0].study_uid == first.study_uid
    assert first_report.patients[0].series_uid == first.series_uid
    destination = output / "PATIENT111"
    assert sorted(path.name for path in destination.iterdir()) == [
        "first000002.dcm",
        "first000003.dcm",
    ]

    selection = _inventory(source, workbook, review, force=True)
    _select(selection, "Patient 1", second.series_uid)
    second_report = curate_dataset(source, output, workbook, selection)
    assert second_report.patients[0].series_uid == second.series_uid
    assert sorted(path.name for path in destination.iterdir()) == [
        "second000002.dcm",
        "second000003.dcm",
    ]
    assert all(item.action == "overwritten" for item in second_report.patients[0].files)


def test_stale_selection_rejected_before_existing_output_changes(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    key = _write_series(source, "PATIENT111", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 111, "0-1")])
    selection = _inventory(source, workbook, tmp_path / "review")
    _select(selection, "Patient 1", key.series_uid)
    output = tmp_path / "curated"
    curated = curate_dataset(source, output, workbook, selection)
    expected = sorted(path.name for path in (output / "PATIENT111").iterdir())
    assert curated.patients[0].status == "copied"

    _write_slice(
        source / "PATIENT111" / "IM999999.dcm",
        series_uid=key.series_uid,
        z=9.0,
        stored_value=9,
        instance_number=9,
    )
    with pytest.raises(ValueError, match="scan selection is stale for Patient 1"):
        curate_dataset(source, output, workbook, selection)
    assert sorted(path.name for path in (output / "PATIENT111").iterdir()) == expected


def test_curation_is_idempotent_forceful_and_records_provenance(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    key = _write_series(source, "PATIENT111", 4)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 111, "0-2")])
    selection = _inventory(source, workbook, tmp_path / "review")
    _select(selection, "Patient 1", key.series_uid)
    output = tmp_path / "curated"

    first = curate_dataset(source, output, workbook, selection)
    second = curate_dataset(source, output, workbook, selection)
    assert first.patients[0].copied_file_count == 3
    assert second.patients[0].unchanged_file_count == 3
    assert [item.action for item in second.patients[0].files] == ["unchanged"] * 3
    forced = curate_dataset(source, output, workbook, selection, force=True)
    assert forced.patients[0].copied_file_count == 3
    assert all(item.action == "overwritten" for item in forced.patients[0].files)

    summary, manifest_path = write_curation_report(forced)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    patient = manifest["patients"][0]
    assert manifest["selection_sha256"]
    assert patient["candidate_id"]
    assert patient["study_uid"] == key.study_uid
    assert patient["series_uid"] == key.series_uid
    assert patient["series_sop_uids_sha256"]
    assert summary.read_text(encoding="utf-8-sig").splitlines()[0].startswith("patient_id,")


def test_subset_filter_accepts_patient_and_folder_aliases(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    one = _write_series(source, "PATIENT111", 3)
    two = _write_series(source, "PATIENT222", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(
        workbook,
        [_row("Patient 1", 111, "0-1"), _row("Patient 2", 222, "0-1")],
    )
    selection = _inventory(source, workbook, tmp_path / "review", patients={"PATIENT222"})
    _select(selection, "Patient 2", two.series_uid)
    output = tmp_path / "curated"
    report = curate_dataset(source, output, workbook, selection, patients={"Patient 2"})
    assert [patient.patient_id for patient in report.patients] == ["Patient 2"]
    assert (output / "PATIENT222").is_dir()
    assert not (output / "PATIENT111").exists()
    assert one != two


def test_inspect_dataset_consumes_curated_one_series_directory(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    key = _write_series(source, "PATIENT111", 4, gap_after=2)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 111, "0-3")])
    selection = _inventory(source, workbook, tmp_path / "review")
    _select(selection, "Patient 1", key.series_uid)
    curated = tmp_path / "curated"
    write_curation_report(curate_dataset(source, curated, workbook, selection))
    inspection = inspect_dataset(curated, workbook_path=workbook)[0]
    assert inspection.study_uid == key.study_uid
    assert inspection.series_uid == key.series_uid
    assert inspection.geometry_status == "eligible"
    assert "gap" in (inspection.reason or "")


def test_inventory_and_preprocess_cli_end_to_end(tmp_path: Path) -> None:
    source = tmp_path / "dicom"
    key = _write_series(source, "PATIENT111", 3)
    workbook = tmp_path / "workbook.xlsx"
    _write_workbook(workbook, [_row("Patient 1", 111, "0-1")])
    review = tmp_path / "review"
    runner = CliRunner()
    inventory_result = runner.invoke(
        app,
        [
            "inventory",
            "--dicom-root",
            str(source),
            "--workbook",
            str(workbook),
            "--output-dir",
            str(review),
        ],
    )
    assert inventory_result.exit_code == 0, inventory_result.output
    assert "1 candidates, 1 ready" in inventory_result.output
    selection = review / "scan_selection.csv"
    _select(selection, "Patient 1", key.series_uid)
    output = tmp_path / "curated"
    preprocess_result = runner.invoke(
        app,
        [
            "preprocess",
            "--dicom-root",
            str(source),
            "--workbook",
            str(workbook),
            "--selection",
            str(selection),
            "--output-dir",
            str(output),
        ],
    )
    assert preprocess_result.exit_code == 0, preprocess_result.output
    assert "Patient 1: copied" in preprocess_result.output
    assert (output / "curation_manifest.json").is_file()


def test_cli_help_describes_review_contract() -> None:
    runner = CliRunner()
    inventory_help = runner.invoke(app, ["inventory", "--help"])
    preprocess_help = runner.invoke(app, ["preprocess", "--help"])
    assert inventory_help.exit_code == 0
    assert "editable CT Series inventory" in inventory_help.output
    assert preprocess_help.exit_code == 0
    assert "Inventory CSV" in preprocess_help.output
