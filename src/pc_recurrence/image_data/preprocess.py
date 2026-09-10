from __future__ import annotations

import csv
import hashlib
import json
import shutil
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pc_recurrence import __version__
from pc_recurrence.io import sha256_file

from .dicom import (
    CT_IMAGE_STORAGE_UID,
    DicomGeometryError,
    DicomSeries,
    discover_dicom_series,
    inspect_patient,
    select_series_files,
)
from .workbook import ImageWorkbookRow, load_image_workbook, select_image_workbook_rows


@dataclass
class CuratedFile:
    source: str
    destination: str
    sha256: str
    action: str

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


@dataclass
class CuratedPatient:
    patient_id: str
    row_number: int
    hasta_no: str | float | None
    dicom_folder: str | None
    image_range_raw: str | None
    status: str
    reason: str | None
    source_dir: str | None
    destination_dir: str | None
    geometry_status: str | None
    geometry_reason: str | None
    study_uid: str | None = None
    series_uid: str | None = None
    series_sop_uids_sha256: str | None = None
    source_file_count: int = 0
    duplicate_file_count: int = 0
    selected_file_count: int = 0
    copied_file_count: int = 0
    unchanged_file_count: int = 0
    selected_instance_numbers: list[int] = field(default_factory=list)
    selected_sop_instance_uids: list[str] = field(default_factory=list)
    files: list[CuratedFile] = field(default_factory=list)
    slices_are_preselected: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["files"] = [item.to_dict() for item in self.files]
        return data


@dataclass
class CurationReport:
    dicom_root: Path
    output_root: Path
    workbook_path: Path
    patients: list[CuratedPatient]

    @property
    def failures(self) -> list[CuratedPatient]:
        return [patient for patient in self.patients if patient.status == "skipped"]

    @property
    def totals(self) -> dict[str, int]:
        return {
            "patient_count": len(self.patients),
            "copied_patient_count": sum(patient.status == "copied" for patient in self.patients),
            "skipped_patient_count": len(self.failures),
            "selected_slice_count": sum(patient.selected_file_count for patient in self.patients),
            "copied_file_count": sum(patient.copied_file_count for patient in self.patients),
            "unchanged_file_count": sum(patient.unchanged_file_count for patient in self.patients),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_version": __version__,
            "stage": "dicom_slice_curation",
            "status": "complete" if not self.failures else "complete_with_skips",
            "dicom_root": str(self.dicom_root.resolve()),
            "output_root": str(self.output_root.resolve()),
            "workbook": str(self.workbook_path.resolve()),
            "workbook_sha256": sha256_file(self.workbook_path),
            "policy": {
                "series_selection": "one processable CT Series per professionally curated folder",
                "duplicate_policy": "byte-identical SOPInstanceUID copies collapse to one file",
                "indexing": (
                    "preselected source slices retained in full"
                    if any(row.slices_are_preselected for row in self.patients)
                    else "zero-based inclusive ordinals after ascending DICOM InstanceNumber"
                ),
                "copy_policy": "staged complete-set replacement with SHA-256 verification",
                "idempotent": True,
                "source_preserved": True,
            },
            "totals": self.totals,
            "patients": [patient.to_dict() for patient in self.patients],
            "failures": [patient.to_dict() for patient in self.failures],
        }


SUMMARY_COLUMNS = (
    "patient_id",
    "row_number",
    "hasta_no",
    "dicom_folder",
    "study_uid",
    "series_uid",
    "series_sop_uids_sha256",
    "source_file_count",
    "duplicate_file_count",
    "status",
    "reason",
    "geometry_status",
    "image_range_raw",
    "selected_file_count",
    "copied_file_count",
    "unchanged_file_count",
    "source_dir",
    "destination_dir",
)


def _series_sop_uids_sha256(series: DicomSeries) -> str:
    values = sorted(str(dataset.SOPInstanceUID) for _, dataset in series.headers)
    return hashlib.sha256("\n".join(values).encode()).hexdigest()


def _skipped_patient(
    row: ImageWorkbookRow,
    *,
    reason: str,
    source_dir: Path,
    destination_dir: Path,
) -> CuratedPatient:
    return CuratedPatient(
        patient_id=row.patient_id,
        row_number=row.row_number,
        hasta_no=row.hasta_no,
        dicom_folder=row.dicom_folder,
        image_range_raw=row.image_range_raw,
        slices_are_preselected=row.slices_are_preselected,
        status="skipped",
        reason=reason,
        source_dir=str(source_dir.resolve()),
        destination_dir=str(destination_dir.resolve()),
        geometry_status=None,
        geometry_reason=None,
    )


def _processable_series(source_dir: Path) -> list[DicomSeries]:
    return [
        series
        for series in discover_dicom_series(source_dir).series
        if series.sop_class_uid == CT_IMAGE_STORAGE_UID and not series.problems
    ]


def _curate_patient(
    output_root: Path,
    row: ImageWorkbookRow,
    source_dir: Path,
    series: DicomSeries,
    *,
    force: bool,
) -> CuratedPatient:
    folder = row.dicom_folder or ""
    destination_dir = output_root / folder
    try:
        files, instance_numbers, sop_uids = select_series_files(
            source_dir, series.key, row.selection_range_raw
        )
    except DicomGeometryError as exc:
        return _skipped_patient(
            row,
            reason=str(exc),
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    basenames = [path.name for path in files]
    if len(basenames) != len(set(basenames)):
        return _skipped_patient(
            row,
            reason="duplicate destination basenames in selected files",
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    desired_hashes = {path.name: sha256_file(path) for path in files}
    destination_matches = False
    if destination_dir.is_dir() and not force:
        entries = list(destination_dir.iterdir())
        existing = {path.name: path for path in entries if path.is_file()}
        destination_matches = (
            len(existing) == len(entries)
            and set(existing) == set(desired_hashes)
            and all(
                sha256_file(existing[name]) == digest for name, digest in desired_hashes.items()
            )
        )
    if destination_matches:
        curated_files = [
            CuratedFile(
                source=str(path),
                destination=str(destination_dir / path.name),
                sha256=desired_hashes[path.name],
                action="unchanged",
            )
            for path in files
        ]
        copied_count = 0
        unchanged_count = len(files)
    else:
        output_root.mkdir(parents=True, exist_ok=True)
        staging_dir = output_root / f".{folder}.staging-{uuid.uuid4().hex}"
        backup_dir = output_root / f".{folder}.backup-{uuid.uuid4().hex}"
        replacement = destination_dir.exists()
        curated_files = []
        try:
            staging_dir.mkdir()
            for path in files:
                staged = staging_dir / path.name
                shutil.copy2(path, staged)
                if sha256_file(staged) != desired_hashes[path.name]:
                    raise RuntimeError(f"SHA-256 mismatch after copy for {path.name}")
            if replacement:
                destination_dir.replace(backup_dir)
            try:
                staging_dir.replace(destination_dir)
            except Exception:
                if backup_dir.exists() and not destination_dir.exists():
                    backup_dir.replace(destination_dir)
                raise
            if backup_dir.is_dir():
                shutil.rmtree(backup_dir)
            elif backup_dir.exists():
                backup_dir.unlink()
            action = "overwritten" if replacement else "copied"
            curated_files = [
                CuratedFile(
                    source=str(path),
                    destination=str(destination_dir / path.name),
                    sha256=desired_hashes[path.name],
                    action=action,
                )
                for path in files
            ]
        except Exception as exc:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
            if backup_dir.exists() and not destination_dir.exists():
                backup_dir.replace(destination_dir)
            return _skipped_patient(
                row,
                reason=f"{type(exc).__name__}: {exc}",
                source_dir=source_dir,
                destination_dir=destination_dir,
            )
        copied_count = len(files)
        unchanged_count = 0

    inspection = inspect_patient(destination_dir, patient_id=row.patient_id, selection=series.key)
    return CuratedPatient(
        patient_id=row.patient_id,
        row_number=row.row_number,
        hasta_no=row.hasta_no,
        dicom_folder=row.dicom_folder,
        image_range_raw=row.image_range_raw,
        slices_are_preselected=row.slices_are_preselected,
        status="copied",
        reason=None,
        source_dir=str(source_dir.resolve()),
        destination_dir=str(destination_dir.resolve()),
        geometry_status=inspection.geometry_status,
        geometry_reason=inspection.reason,
        study_uid=series.key.study_uid,
        series_uid=series.key.series_uid,
        series_sop_uids_sha256=_series_sop_uids_sha256(series),
        source_file_count=series.source_file_count,
        duplicate_file_count=series.duplicate_file_count,
        selected_file_count=len(files),
        copied_file_count=copied_count,
        unchanged_file_count=unchanged_count,
        selected_instance_numbers=list(instance_numbers),
        selected_sop_instance_uids=list(sop_uids),
        files=curated_files,
    )


def _curate_row(
    dicom_root: Path,
    output_root: Path,
    row: ImageWorkbookRow,
    *,
    force: bool,
) -> CuratedPatient:
    folder = row.dicom_folder or ""
    source_dir = dicom_root / folder
    destination_dir = output_root / folder
    if not folder or not source_dir.is_dir():
        return _skipped_patient(
            row,
            reason="professionally curated CT folder is missing",
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    series = _processable_series(source_dir)
    if len(series) != 1:
        reason = (
            "no processable CT DICOM Series found"
            if not series
            else f"expected one processable CT DICOM Series; found {len(series)}"
        )
        return _skipped_patient(
            row,
            reason=reason,
            source_dir=source_dir,
            destination_dir=destination_dir,
        )
    return _curate_patient(output_root, row, source_dir, series[0], force=force)


def curate_dataset(
    dicom_root: Path,
    output_root: Path,
    workbook_path: Path,
    *,
    patients: set[str] | None = None,
    force: bool = False,
) -> CurationReport:
    """Synchronize one professionally selected CT Series per workbook case."""
    workbook_rows = select_image_workbook_rows(load_image_workbook(workbook_path), patients)
    report_patients = [
        _curate_row(dicom_root, output_root, row, force=force) for row in workbook_rows
    ]
    return CurationReport(
        dicom_root=dicom_root,
        output_root=output_root,
        workbook_path=workbook_path,
        patients=report_patients,
    )


def write_curation_report(report: CurationReport) -> tuple[Path, Path]:
    report.output_root.mkdir(parents=True, exist_ok=True)
    summary_path = report.output_root / "curation_summary.csv"
    manifest_path = report.output_root / "curation_manifest.json"
    temporary_summary = summary_path.with_suffix(".csv.tmp")
    with temporary_summary.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(patient.to_dict() for patient in report.patients)
    temporary_summary.replace(summary_path)
    manifest_path.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary_path, manifest_path
