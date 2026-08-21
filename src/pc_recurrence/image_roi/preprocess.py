from __future__ import annotations

import csv
import hashlib
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pc_recurrence import __version__

from .dicom import DicomGeometryError, inspect_patient, select_series_files
from .pipeline import workbook_entries
from .workbook import load_image_workbook


@dataclass
class CuratedFile:
    source: str
    destination: str
    sha256: str
    action: str  # "copied" | "unchanged" | "overwritten"

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


@dataclass
class CuratedPatient:
    patient_id: str
    row_number: int
    hasta_no: str | float | None
    dicom_folder: str | None
    image_range_raw: str | None
    status: str  # "copied" | "skipped"
    reason: str | None
    source_dir: str | None
    destination_dir: str | None
    geometry_status: str | None
    geometry_reason: str | None
    selected_file_count: int = 0
    copied_file_count: int = 0
    unchanged_file_count: int = 0
    selected_instance_numbers: list[int] = field(default_factory=list)
    selected_sop_instance_uids: list[str] = field(default_factory=list)
    files: list[CuratedFile] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "patient_id": self.patient_id,
            "row_number": self.row_number,
            "hasta_no": self.hasta_no,
            "dicom_folder": self.dicom_folder,
            "image_range_raw": self.image_range_raw,
            "status": self.status,
            "reason": self.reason,
            "source_dir": self.source_dir,
            "destination_dir": self.destination_dir,
            "geometry_status": self.geometry_status,
            "geometry_reason": self.geometry_reason,
            "selected_file_count": self.selected_file_count,
            "copied_file_count": self.copied_file_count,
            "unchanged_file_count": self.unchanged_file_count,
            "selected_instance_numbers": self.selected_instance_numbers,
            "selected_sop_instance_uids": self.selected_sop_instance_uids,
            "files": [item.to_dict() for item in self.files],
        }


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
            "selected_slice_count": sum(
                patient.selected_file_count for patient in self.patients
            ),
            "copied_file_count": sum(
                patient.copied_file_count for patient in self.patients
            ),
            "unchanged_file_count": sum(
                patient.unchanged_file_count for patient in self.patients
            ),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_version": __version__,
            "stage": "dicom_slice_curation",
            "status": "complete" if not self.failures else "complete_with_skips",
            "dicom_root": str(self.dicom_root.resolve()),
            "output_root": str(self.output_root.resolve()),
            "workbook": str(self.workbook_path.resolve()),
            "workbook_sha256": _file_sha256(self.workbook_path),
            "policy": {
                "indexing": "one-based inclusive ordinals after ascending DICOM InstanceNumber",
                "copy_policy": "shutil.copy2 with SHA-256 verification",
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _skipped_patient(
    *,
    patient_id: str,
    row_number: int,
    hasta_no: str | float | None,
    dicom_folder: str | None,
    image_range_raw: str | None,
    reason: str,
    source_dir: str | None,
    destination_dir: str | None = None,
    geometry_status: str | None = None,
    geometry_reason: str | None = None,
) -> CuratedPatient:
    return CuratedPatient(
        patient_id=patient_id,
        row_number=row_number,
        hasta_no=hasta_no,
        dicom_folder=dicom_folder,
        image_range_raw=image_range_raw,
        status="skipped",
        reason=reason,
        source_dir=source_dir,
        destination_dir=destination_dir,
        geometry_status=geometry_status,
        geometry_reason=geometry_reason,
    )


def _curate_patient(
    dicom_root: Path,
    output_root: Path,
    *,
    patient_id: str,
    row_number: int,
    hasta_no: str | float | None,
    dicom_folder: str | None,
    image_range_raw: str | None,
    force: bool,
) -> CuratedPatient:
    source_dir = dicom_root / dicom_folder if dicom_folder else None
    source_dir_text = str(source_dir.resolve()) if source_dir else None
    if source_dir is None or not source_dir.is_dir():
        return _skipped_patient(
            patient_id=patient_id,
            row_number=row_number,
            hasta_no=hasta_no,
            dicom_folder=dicom_folder,
            image_range_raw=image_range_raw,
            reason="no source DICOM folder",
            source_dir=source_dir_text,
        )
    if not isinstance(image_range_raw, str) or not image_range_raw.strip():
        return _skipped_patient(
            patient_id=patient_id,
            row_number=row_number,
            hasta_no=hasta_no,
            dicom_folder=dicom_folder,
            image_range_raw=image_range_raw,
            reason="no workbook image range",
            source_dir=source_dir_text,
        )

    try:
        files, instance_numbers, sop_uids = select_series_files(source_dir, image_range_raw)
    except DicomGeometryError as exc:
        return _skipped_patient(
            patient_id=patient_id,
            row_number=row_number,
            hasta_no=hasta_no,
            dicom_folder=dicom_folder,
            image_range_raw=image_range_raw,
            reason=str(exc),
            source_dir=source_dir_text,
        )

    destination_dir = output_root / dicom_folder
    destination_dir.mkdir(parents=True, exist_ok=True)
    curated_files: list[CuratedFile] = []
    copied_count = 0
    unchanged_count = 0
    try:
        for path in files:
            destination = destination_dir / path.name
            source_sha256 = _file_sha256(path)
            existed = destination.exists()
            if existed and not force and _file_sha256(destination) == source_sha256:
                unchanged_count += 1
                action = "unchanged"
            else:
                shutil.copy2(path, destination)
                copied_count += 1
                action = "overwritten" if existed else "copied"
            verified_sha256 = _file_sha256(destination)
            if verified_sha256 != source_sha256:
                raise RuntimeError(
                    f"SHA-256 mismatch after copy for {path.name}"
                )
            curated_files.append(
                CuratedFile(
                    source=str(path),
                    destination=str(destination),
                    sha256=verified_sha256,
                    action=action,
                )
            )
    except Exception as exc:
        return _skipped_patient(
            patient_id=patient_id,
            row_number=row_number,
            hasta_no=hasta_no,
            dicom_folder=dicom_folder,
            image_range_raw=image_range_raw,
            reason=f"{type(exc).__name__}: {exc}",
            source_dir=source_dir_text,
            destination_dir=str(destination_dir.resolve()),
        )

    inspection = inspect_patient(destination_dir, patient_id=patient_id)
    return CuratedPatient(
        patient_id=patient_id,
        row_number=row_number,
        hasta_no=hasta_no,
        dicom_folder=dicom_folder,
        image_range_raw=image_range_raw,
        status="copied",
        reason=None,
        source_dir=source_dir_text,
        destination_dir=str(destination_dir.resolve()),
        geometry_status=inspection.geometry_status,
        geometry_reason=inspection.reason,
        selected_file_count=len(files),
        copied_file_count=copied_count,
        unchanged_file_count=unchanged_count,
        selected_instance_numbers=list(instance_numbers),
        selected_sop_instance_uids=list(sop_uids),
        files=curated_files,
    )


def curate_dataset(
    dicom_root: Path,
    output_root: Path,
    workbook_path: Path,
    *,
    patients: set[str] | None = None,
    force: bool = False,
) -> CurationReport:
    """Copy workbook-defined slice ranges into a curated DICOM folder.

    The workbook is processed in table order. Patients whose DICOM folder is
    missing, whose range is missing or invalid, or whose copy verification
    fails are recorded as skips and never abort the run. Geometry eligibility
    is documented per patient but does not gate the copy, so a table range
    spanning two acquisition blocks (e.g. PATIENT853534) is copied as-is.
    Re-running skips byte-identical files unless ``force`` is set.
    """
    entries = workbook_entries(workbook_path)
    output_root.mkdir(parents=True, exist_ok=True)
    report_patients: list[CuratedPatient] = []
    for row in load_image_workbook(workbook_path):
        patient_id = row.patient_id
        dicom_folder = row.dicom_folder
        if patients is not None and patient_id not in patients and dicom_folder not in patients:
            continue
        report_patients.append(
            _curate_patient(
                dicom_root,
                output_root,
                patient_id=patient_id,
                row_number=row.row_number,
                hasta_no=row.hasta_no,
                dicom_folder=dicom_folder,
                image_range_raw=entries[patient_id]["image_range_raw"],
                force=force,
            )
        )
    return CurationReport(
        dicom_root=dicom_root,
        output_root=output_root,
        workbook_path=workbook_path,
        patients=report_patients,
    )


def write_curation_report(report: CurationReport) -> tuple[Path, Path]:
    """Write curation_summary.csv and curation_manifest.json into the output root."""
    summary_path = report.output_root / "curation_summary.csv"
    with summary_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for patient in report.patients:
            writer.writerow({column: getattr(patient, column) for column in SUMMARY_COLUMNS})
    manifest_path = report.output_root / "curation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(report.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary.replace(manifest_path)
    return summary_path, manifest_path
