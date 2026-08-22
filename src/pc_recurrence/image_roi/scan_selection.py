from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pc_recurrence import __version__

from .artifacts import render_dicom_series_preview, write_json
from .dicom import (
    CT_IMAGE_STORAGE_UID,
    DicomGeometryError,
    DicomSeries,
    SeriesKey,
    _geometry,
    _select_instance_range,
    discover_dicom_series,
)
from .workbook import ImageWorkbookRow, load_image_workbook, select_image_workbook_rows

SCAN_SELECTION_COLUMNS = (
    "selected",
    "patient_id",
    "dicom_folder",
    "candidate_id",
    "status",
    "reason",
    "study_instance_uid",
    "series_instance_uid",
    "sop_class_uid",
    "source_directories",
    "source_file_count",
    "unique_file_count",
    "duplicate_file_count",
    "series_sop_uids_sha256",
    "study_date",
    "study_description",
    "series_number",
    "acquisition_number",
    "series_description",
    "protocol_name",
    "body_part_examined",
    "contrast_bolus_agent",
    "rows",
    "columns",
    "slice_thickness_mm",
    "row_spacing_mm",
    "column_spacing_mm",
    "image_range",
    "selected_file_count",
    "median_slice_spacing_mm",
    "maximum_slice_gap_mm",
    "geometry_warnings",
    "preview_status",
    "preview_reason",
    "preview_path",
)


@dataclass(frozen=True)
class PatientSeriesSelection:
    patient_id: str
    dicom_folder: str
    candidate_id: str
    key: SeriesKey


class ScanSelectionConflictError(RuntimeError):
    pass


@dataclass(frozen=True)
class ScanSelectionDocument:
    selection_path: Path
    selection_sha256: str
    rows: list[dict[str, str]]
    patient_ids: tuple[str, ...]
    selected_candidate_ids: dict[str, str | None]


@dataclass
class ScanInventoryReport:
    dicom_root: Path
    workbook_path: Path
    output_dir: Path
    rows: list[dict[str, Any]]
    patient_summaries: list[dict[str, Any]]
    selection_path: Path
    manifest_path: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def candidate_id(key: SeriesKey) -> str:
    return hashlib.sha256(f"{key.study_uid}\n{key.series_uid}".encode()).hexdigest()


def series_sop_uids_sha256(series: DicomSeries) -> str:
    values = sorted({str(dataset.SOPInstanceUID) for _, dataset in series.headers})
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _text(dataset: Any, name: str) -> str:
    value = getattr(dataset, name, "")
    return str(value).strip() if value is not None else ""


def _candidate_row(
    workbook_row: ImageWorkbookRow,
    series: DicomSeries,
) -> tuple[dict[str, Any], list[Path]]:
    first = series.headers[0][1]
    status = "ready"
    reasons = list(series.problems)
    selected_headers = series.headers
    if series.sop_class_uid != CT_IMAGE_STORAGE_UID:
        reasons.append(f"unsupported SOPClassUID {series.sop_class_uid or '<missing>'}")
    if workbook_row.image_range_raw is None:
        reasons.append("missing image range")
    if not reasons:
        try:
            selected_headers, _, _ = _select_instance_range(
                series.headers, workbook_row.image_range_raw
            )
        except DicomGeometryError as exc:
            reasons.append(str(exc))
    if reasons:
        status = "not_selectable"
        selected_headers = series.headers

    geometry_warnings: tuple[str, ...] = ()
    median_spacing: float | str = ""
    maximum_gap: float | str = ""
    preview_files = [path for path, _ in selected_headers]
    try:
        (
            ordered,
            _,
            _,
            _,
            median_spacing,
            maximum_gap,
            _,
            _,
            geometry_warnings,
        ) = _geometry(selected_headers)
        preview_files = [path for path, _ in ordered]
    except Exception as exc:
        if status == "ready":
            status = "not_selectable"
            reasons.append(str(exc))

    pixel_spacing = getattr(first, "PixelSpacing", ())
    try:
        row_spacing = float(pixel_spacing[0])
        column_spacing = float(pixel_spacing[1])
    except (IndexError, TypeError, ValueError):
        row_spacing = ""
        column_spacing = ""
    key = series.key
    row: dict[str, Any] = {
        "selected": "",
        "patient_id": workbook_row.patient_id,
        "dicom_folder": workbook_row.dicom_folder or "",
        "candidate_id": candidate_id(key),
        "status": status,
        "reason": "; ".join(reasons),
        "study_instance_uid": key.study_uid,
        "series_instance_uid": key.series_uid,
        "sop_class_uid": series.sop_class_uid,
        "source_directories": json.dumps(series.source_directories, ensure_ascii=False),
        "source_file_count": series.source_file_count,
        "unique_file_count": len(series.headers),
        "duplicate_file_count": series.duplicate_file_count,
        "series_sop_uids_sha256": series_sop_uids_sha256(series),
        "study_date": _text(first, "StudyDate"),
        "study_description": _text(first, "StudyDescription"),
        "series_number": _text(first, "SeriesNumber"),
        "acquisition_number": _text(first, "AcquisitionNumber"),
        "series_description": _text(first, "SeriesDescription"),
        "protocol_name": _text(first, "ProtocolName"),
        "body_part_examined": _text(first, "BodyPartExamined"),
        "contrast_bolus_agent": _text(first, "ContrastBolusAgent"),
        "rows": _text(first, "Rows"),
        "columns": _text(first, "Columns"),
        "slice_thickness_mm": _text(first, "SliceThickness"),
        "row_spacing_mm": row_spacing,
        "column_spacing_mm": column_spacing,
        "image_range": workbook_row.image_range_raw or "",
        "selected_file_count": len(selected_headers) if status == "ready" else "",
        "median_slice_spacing_mm": median_spacing,
        "maximum_slice_gap_mm": maximum_gap,
        "geometry_warnings": "; ".join(geometry_warnings),
        "preview_status": "",
        "preview_reason": "",
        "preview_path": "",
    }
    return row, preview_files


def _no_series_row(row: ImageWorkbookRow, reason: str) -> dict[str, Any]:
    result = {column: "" for column in SCAN_SELECTION_COLUMNS}
    result.update(
        {
            "patient_id": row.patient_id,
            "dicom_folder": row.dicom_folder or "",
            "status": "no_series",
            "reason": reason,
            "image_range": row.image_range_raw or "",
        }
    )
    return result


def _live_rows(
    dicom_root: Path,
    workbook_rows: list[ImageWorkbookRow],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[tuple[str, str], tuple[DicomSeries, ImageWorkbookRow, list[Path]]],
]:
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    candidates: dict[tuple[str, str], tuple[DicomSeries, ImageWorkbookRow, list[Path]]] = {}
    for workbook_row in workbook_rows:
        folder = workbook_row.dicom_folder or ""
        patient_dir = dicom_root / folder if folder else None
        if patient_dir is None or not patient_dir.is_dir():
            reason = "patient folder is missing; ignored 0 files"
            rows.append(_no_series_row(workbook_row, reason))
            summaries.append(
                {
                    "patient_id": workbook_row.patient_id,
                    "dicom_folder": folder,
                    "candidate_count": 0,
                    "ready_count": 0,
                    "issues": [reason],
                }
            )
            continue
        discovery = discover_dicom_series(patient_dir)
        if not discovery.series:
            reason = f"no DICOM CT Series found; ignored {discovery.ignored_file_count} files"
            rows.append(_no_series_row(workbook_row, reason))
            summaries.append(
                {
                    "patient_id": workbook_row.patient_id,
                    "dicom_folder": folder,
                    "candidate_count": 0,
                    "ready_count": 0,
                    "issues": [reason],
                }
            )
            continue
        patient_rows: list[dict[str, Any]] = []
        for series in discovery.series:
            candidate_row, preview_files = _candidate_row(workbook_row, series)
            patient_rows.append(candidate_row)
            candidates[(workbook_row.patient_id, candidate_row["candidate_id"])] = (
                series,
                workbook_row,
                preview_files,
            )
        rows.extend(patient_rows)
        summaries.append(
            {
                "patient_id": workbook_row.patient_id,
                "dicom_folder": folder,
                "candidate_count": len(patient_rows),
                "ready_count": sum(item["status"] == "ready" for item in patient_rows),
                "issues": list(discovery.issues),
            }
        )
    return rows, summaries, candidates


def _write_csv_atomic(
    rows: list[dict[str, Any]],
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> str:
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=SCAN_SELECTION_COLUMNS,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(rows)
        revision = _sha256(temporary)
        if expected_sha256 is not None:
            current_revision = _sha256(path) if path.exists() else ""
            if current_revision != expected_sha256:
                raise ScanSelectionConflictError(
                    "scan selection changed on disk; reload before saving"
                )
        temporary.replace(path)
        return revision
    finally:
        temporary.unlink(missing_ok=True)


def write_scan_inventory(
    dicom_root: Path,
    workbook_path: Path,
    output_dir: Path,
    *,
    patients: set[str] | None = None,
    force: bool = False,
) -> ScanInventoryReport:
    selection_path = output_dir / "scan_selection.csv"
    manifest_path = output_dir / "scan_inventory_manifest.json"
    if selection_path.exists() and not force:
        raise FileExistsError(f"Refusing to overwrite existing scan selection: {selection_path}")
    workbook_rows = select_image_workbook_rows(load_image_workbook(workbook_path), patients)
    live_rows, summaries, candidates = _live_rows(dicom_root, workbook_rows)

    if force and output_dir.exists():
        previews = output_dir / "previews"
        if previews.exists():
            shutil.rmtree(previews)
    output_dir.mkdir(parents=True, exist_ok=True)
    for record in live_rows:
        if not record["candidate_id"]:
            continue
        series, workbook_row, preview_files = candidates[
            (record["patient_id"], record["candidate_id"])
        ]
        relative_path = (
            Path("previews")
            / (workbook_row.dicom_folder or workbook_row.patient_id)
            / f"{record['candidate_id']}.png"
        )
        title = (
            f"{workbook_row.patient_id} / {workbook_row.dicom_folder or ''}\n"
            f"Sources: {', '.join(series.source_directories)}\n"
            f"Study: {series.key.study_uid}\nSeries: {series.key.series_uid}\n"
            f"Range: {record['image_range']} | Status: {record['status']}\n"
            f"Geometry: {record['geometry_warnings'] or 'none'}"
        )
        try:
            render_dicom_series_preview(preview_files, output_dir / relative_path, title=title)
            record["preview_status"] = "ready"
            record["preview_path"] = relative_path.as_posix()
        except Exception as exc:
            record["preview_status"] = "failed"
            record["preview_reason"] = str(exc)

    _write_csv_atomic(live_rows, selection_path)
    manifest = {
        "pipeline_version": __version__,
        "stage": "ct_series_inventory",
        "dicom_root": str(dicom_root.resolve()),
        "workbook_path": str(workbook_path.resolve()),
        "workbook_sha256": _sha256(workbook_path),
        "output_dir": str(output_dir.resolve()),
        "selection_path": str(selection_path.resolve()),
        "patient_count": len(workbook_rows),
        "candidate_count": sum(bool(row["candidate_id"]) for row in live_rows),
        "status_counts": {
            status: sum(row["status"] == status for row in live_rows)
            for status in ("ready", "not_selectable", "no_series")
        },
        "patient_summaries": summaries,
        "candidates": live_rows,
        "artifacts": {
            "selection_csv": selection_path.name,
            "previews": "previews",
        },
    }
    write_json(manifest, manifest_path)
    return ScanInventoryReport(
        dicom_root=dicom_root,
        workbook_path=workbook_path,
        output_dir=output_dir,
        rows=live_rows,
        patient_summaries=summaries,
        selection_path=selection_path,
        manifest_path=manifest_path,
    )


def read_scan_selection(selection_path: Path) -> ScanSelectionDocument:
    file_bytes = selection_path.read_bytes()
    revision = hashlib.sha256(file_bytes).hexdigest()
    text = file_bytes.decode("utf-8-sig")
    reader = csv.DictReader(io.StringIO(text, newline=""))
    fieldnames = tuple(reader.fieldnames or ())
    if fieldnames != SCAN_SELECTION_COLUMNS:
        raise ValueError(
            f"Unexpected scan selection schema. Expected {SCAN_SELECTION_COLUMNS}; "
            f"found {fieldnames}"
        )
    rows = list(reader)
    if not rows:
        raise ValueError("scan selection must contain at least one row")

    patient_ids: list[str] = []
    patient_folders: dict[str, str] = {}
    selected_candidate_ids: dict[str, str | None] = {}
    seen_candidates: set[str] = set()
    for row_number, row in enumerate(rows, start=2):
        patient_id = row["patient_id"].strip()
        if not patient_id:
            raise ValueError(f"blank patient_id in scan selection row {row_number}")
        if patient_id not in patient_folders:
            patient_ids.append(patient_id)
            patient_folders[patient_id] = row["dicom_folder"]
            selected_candidate_ids[patient_id] = None
        elif row["dicom_folder"] != patient_folders[patient_id]:
            raise ValueError(f"inconsistent DICOM folder for patient {patient_id}")

        status = row["status"]
        if status not in {"ready", "not_selectable", "no_series"}:
            raise ValueError(
                f"invalid status {status!r} for patient {patient_id} in row {row_number}"
            )
        identifier = row["candidate_id"]
        if identifier:
            if re.fullmatch(r"[0-9a-f]{64}", identifier) is None:
                raise ValueError(f"malformed candidate_id {identifier!r} for patient {patient_id}")
            if identifier in seen_candidates:
                raise ValueError(f"duplicate candidate {identifier} for patient {patient_id}")
            seen_candidates.add(identifier)

        selected = row["selected"].strip().lower()
        if selected not in {"", "yes"}:
            raise ValueError(
                f"invalid selected value {row['selected']!r} for patient {patient_id}; "
                "expected blank or yes"
            )
        if selected == "yes":
            if selected_candidate_ids[patient_id] is not None:
                raise ValueError(f"multiple selected rows for patient {patient_id}")
            if not identifier:
                raise ValueError(f"selected row has no candidate_id for patient {patient_id}")
            if status != "ready":
                raise ValueError(
                    f"selected candidate {identifier} is not ready for patient {patient_id}"
                )
            selected_candidate_ids[patient_id] = identifier

    return ScanSelectionDocument(
        selection_path=selection_path,
        selection_sha256=revision,
        rows=rows,
        patient_ids=tuple(patient_ids),
        selected_candidate_ids=selected_candidate_ids,
    )


def update_scan_selections(
    selection_path: Path,
    selections: dict[str, str | None],
    *,
    expected_sha256: str,
) -> str:
    document = read_scan_selection(selection_path)
    if document.selection_sha256 != expected_sha256:
        raise ScanSelectionConflictError("scan selection changed on disk; reload before saving")

    expected_patients = set(document.patient_ids)
    submitted_patients = set(selections)
    if submitted_patients != expected_patients:
        missing = sorted(expected_patients - submitted_patients)
        unexpected = sorted(submitted_patients - expected_patients)
        details = []
        if missing:
            details.append(f"missing patients: {', '.join(missing)}")
        if unexpected:
            details.append(f"unknown patients: {', '.join(unexpected)}")
        raise ValueError("selections must include every patient; " + "; ".join(details))

    candidates = {row["candidate_id"]: row for row in document.rows if row["candidate_id"]}
    for patient_id, identifier in selections.items():
        if identifier is None:
            continue
        row = candidates.get(identifier)
        if row is None:
            raise ValueError(f"unknown candidate {identifier} for patient {patient_id}")
        if row["patient_id"].strip() != patient_id:
            raise ValueError(f"candidate {identifier} does not belong to patient {patient_id}")
        if not row["study_instance_uid"] or not row["series_instance_uid"]:
            raise ValueError(
                f"candidate {identifier} has incomplete Study/Series identity "
                f"for patient {patient_id}"
            )
        if row["status"] != "ready":
            raise ValueError(f"candidate {identifier} is not ready for patient {patient_id}")

    updated_rows = [row.copy() for row in document.rows]
    for row in updated_rows:
        row["selected"] = (
            "yes"
            if selections[row["patient_id"].strip()] == row["candidate_id"]
            and bool(row["candidate_id"])
            else ""
        )
    return _write_csv_atomic(
        updated_rows,
        selection_path,
        expected_sha256=document.selection_sha256,
    )


def load_scan_selections(
    selection_path: Path,
    dicom_root: Path,
    workbook_path: Path,
    *,
    patients: set[str] | None = None,
) -> dict[str, PatientSeriesSelection]:
    document = read_scan_selection(selection_path)
    workbook_rows_all = load_image_workbook(workbook_path)
    workbook_rows = select_image_workbook_rows(workbook_rows_all, patients)
    workbook_by_id = {row.patient_id: row for row in workbook_rows_all}
    errors = [
        f"scan selection row does not match workbook: {row['patient_id']}"
        for row in document.rows
        if (
            (workbook_row := workbook_by_id.get(row["patient_id"])) is None
            or row["dicom_folder"] != (workbook_row.dicom_folder or "")
        )
    ]

    live_rows, _, _ = _live_rows(dicom_root, workbook_rows)
    csv_by_patient = {
        row.patient_id: [saved for saved in document.rows if saved["patient_id"] == row.patient_id]
        for row in workbook_rows
    }
    live_by_patient = {
        row.patient_id: [live for live in live_rows if live["patient_id"] == row.patient_id]
        for row in workbook_rows
    }
    stale_fields = (
        "study_instance_uid",
        "series_instance_uid",
        "image_range",
        "source_file_count",
        "unique_file_count",
        "duplicate_file_count",
        "series_sop_uids_sha256",
    )
    selections: dict[str, PatientSeriesSelection] = {}
    for workbook_row in workbook_rows:
        patient_id = workbook_row.patient_id
        patient_csv = csv_by_patient[patient_id]
        patient_live = live_by_patient[patient_id]
        csv_candidates = {row["candidate_id"]: row for row in patient_csv if row["candidate_id"]}
        live_candidates = {row["candidate_id"]: row for row in patient_live if row["candidate_id"]}
        stale = len(patient_csv) != len(patient_live) or set(csv_candidates) != set(live_candidates)
        if not stale:
            stale = any(
                csv_candidates[identifier][field] != str(live[field])
                for identifier, live in live_candidates.items()
                for field in stale_fields
            )
        if not stale and not live_candidates and patient_live:
            stale = any(
                patient_csv[0][field] != str(patient_live[0][field]) for field in stale_fields
            )
        if stale:
            errors.append(f"scan selection is stale for {patient_id}; rerun pc-image-roi inventory")
            continue

        identifier = document.selected_candidate_ids.get(patient_id)
        if identifier is None:
            errors.append(f"{patient_id} requires exactly one selected=yes row; found 0")
            continue
        live = live_candidates.get(identifier)
        if live is None or live["status"] != "ready":
            errors.append(f"selected Series is not ready for {patient_id}")
            continue
        selections[patient_id] = PatientSeriesSelection(
            patient_id=patient_id,
            dicom_folder=workbook_row.dicom_folder or "",
            candidate_id=identifier,
            key=SeriesKey(live["study_instance_uid"], live["series_instance_uid"]),
        )
    if errors:
        raise ValueError("\n".join(errors))
    return selections
