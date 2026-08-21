from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pydicom


class DicomGeometryError(ValueError):
    """Raised when a CT series cannot be represented as one regular volume."""


@dataclass(frozen=True)
class SeriesInspection:
    patient_id: str
    patient_dir: Path
    series_uid: str | None
    file_count: int
    shape: tuple[int, int, int] | None
    spacing_mm: tuple[float, float, float] | None
    median_slice_spacing_mm: float | None
    maximum_slice_gap_mm: float | None
    geometry_status: str
    reason: str | None
    phase_status: str = "phase_unverified"
    selected_instance_numbers: tuple[int, ...] = ()
    selected_sop_instance_uids: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        data = self.__dict__.copy()
        data["patient_dir"] = str(self.patient_dir.resolve())
        data["shape"] = list(self.shape) if self.shape else None
        data["spacing_mm"] = list(self.spacing_mm) if self.spacing_mm else None
        data["selected_instance_numbers"] = list(self.selected_instance_numbers)
        data["selected_sop_instance_uids"] = list(self.selected_sop_instance_uids)
        return data


@dataclass
class DicomVolume:
    patient_id: str
    series_uid: str
    volume_hu: np.ndarray
    affine_ras: np.ndarray
    spacing_mm: tuple[float, float, float]
    files: list[Path]
    median_slice_spacing_mm: float
    maximum_slice_gap_mm: float
    phase_status: str = "phase_unverified"
    selected_instance_numbers: tuple[int, ...] = ()
    selected_sop_instance_uids: tuple[str, ...] = ()

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(int(value) for value in self.volume_hu.shape)


def natural_patient_key(path: Path | str) -> tuple[Any, ...]:
    text = path.name if isinstance(path, Path) else str(path)
    return tuple(int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text))


def patient_directories(dicom_root: Path) -> list[Path]:
    return sorted((path for path in dicom_root.iterdir() if path.is_dir()), key=natural_patient_key)


def _read_headers(patient_dir: Path) -> dict[str, list[tuple[Path, Any]]]:
    groups: dict[str, list[tuple[Path, Any]]] = {}
    for path in sorted(item for item in patient_dir.rglob("*") if item.is_file()):
        try:
            dataset = pydicom.dcmread(path, stop_before_pixels=True, force=True)
        except Exception:
            continue
        if str(getattr(dataset, "Modality", "")).upper() != "CT":
            continue
        uid = str(getattr(dataset, "SeriesInstanceUID", "missing-series-uid"))
        groups.setdefault(uid, []).append((path, dataset))
    return groups


def _largest_ct_series(patient_dir: Path) -> tuple[str, list[tuple[Path, Any]]]:
    groups = _read_headers(patient_dir)
    if not groups:
        raise DicomGeometryError("no CT DICOM series found")
    return max(groups.items(), key=lambda item: (len(item[1]), item[0]))


def parse_image_range(value: str) -> tuple[int, int]:
    """Parse a one-based inclusive workbook slice range."""
    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
    if match is None:
        raise DicomGeometryError(f"invalid image range {value!r}; expected START-STOP")
    start, stop = (int(part) for part in match.groups())
    if start < 1 or stop < start:
        raise DicomGeometryError(
            f"invalid image range {value!r}; indices are one-based and inclusive"
        )
    return start, stop


def _select_instance_range(
    headers: list[tuple[Path, Any]], image_range_raw: str | None
) -> tuple[list[tuple[Path, Any]], tuple[int, ...], tuple[str, ...]]:
    if image_range_raw is None:
        selected = headers
    else:
        start, stop = parse_image_range(image_range_raw)
        numbered: list[tuple[int, Path, Any]] = []
        for path, dataset in headers:
            try:
                instance_number = int(dataset.InstanceNumber)
            except Exception as exc:
                raise DicomGeometryError(
                    f"missing or invalid InstanceNumber in {path.name}"
                ) from exc
            numbered.append((instance_number, path, dataset))
        numbers = [item[0] for item in numbered]
        if len(numbers) != len(set(numbers)):
            raise DicomGeometryError("duplicate InstanceNumber values prevent range mapping")
        numbered.sort(key=lambda item: item[0])
        if stop > len(numbered):
            raise DicomGeometryError(
                f"image range {image_range_raw!r} exceeds {len(numbered)} series slices"
            )
        selected = [(path, dataset) for _, path, dataset in numbered[start - 1 : stop]]
    if len(selected) < 2:
        raise DicomGeometryError("selected image range must contain at least two CT slices")
    try:
        instance_numbers = tuple(int(dataset.InstanceNumber) for _, dataset in selected)
    except Exception as exc:
        if image_range_raw is not None:
            raise DicomGeometryError(
                "selected slices must contain valid InstanceNumber values"
            ) from exc
        instance_numbers = ()
    sop_uids = tuple(str(getattr(dataset, "SOPInstanceUID", "")) for _, dataset in selected)
    if any(not value for value in sop_uids):
        raise DicomGeometryError("selected slices must contain SOPInstanceUID values")
    return selected, instance_numbers, sop_uids


def select_series_files(
    patient_dir: Path,
    image_range_raw: str | None = None,
) -> tuple[list[Path], tuple[int, ...], tuple[str, ...]]:
    """Select the largest CT series files for a patient directory.

    Uses the exact segmentation-pipeline policy: one-based inclusive ordinals
    after ascending DICOM InstanceNumber. Returns the selected file paths in
    instance order, their instance numbers, and their SOP Instance UIDs.
    """
    _, headers = _largest_ct_series(patient_dir)
    selected, instance_numbers, sop_uids = _select_instance_range(headers, image_range_raw)
    return [path for path, _ in selected], instance_numbers, sop_uids


def _geometry(
    headers: list[tuple[Path, Any]],
) -> tuple[
    list[tuple[Path, Any]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float,
    float,
    float,
    float,
    tuple[str, ...],
]:
    """Order slices and derive affine metadata without rejecting anomalies.

    Every slice is accepted: slice positions are sorted stably (duplicates are
    kept), gaps are allowed, and missing geometry tags fall back to safe
    defaults. Anomalies are returned as warnings instead of raising.
    """
    first = headers[0][1]
    orientation = np.asarray(getattr(first, "ImageOrientationPatient", []), dtype=np.float64)
    if orientation.shape != (6,):
        orientation = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    pixel_spacing = np.asarray(getattr(first, "PixelSpacing", []), dtype=np.float64)
    if pixel_spacing.shape != (2,) or np.any(pixel_spacing <= 0):
        pixel_spacing = np.array([1.0, 1.0])
    column_direction = orientation[:3]
    row_direction = orientation[3:]
    slice_normal = np.cross(column_direction, row_direction)
    norm = float(np.linalg.norm(slice_normal))
    if norm <= 0:
        row_direction = np.array([1.0, 0.0, 0.0])
        column_direction = np.array([0.0, 1.0, 0.0])
        slice_normal = np.array([0.0, 0.0, 1.0])
    else:
        slice_normal /= norm

    fallback_spacing = float(getattr(first, "SliceThickness", 1.0) or 1.0)
    missing_positions = any(
        np.asarray(getattr(dataset, "ImagePositionPatient", []), dtype=np.float64).shape != (3,)
        for _, dataset in headers
    )
    located: list[tuple[float, Path, Any]] = []
    for index, (path, dataset) in enumerate(headers):
        if missing_positions:
            coordinate = index * fallback_spacing
        else:
            position = np.asarray(dataset.ImagePositionPatient, dtype=np.float64)
            coordinate = float(np.dot(position, slice_normal))
        located.append((coordinate, path, dataset))
    located.sort(key=lambda item: item[0])
    coordinates = np.asarray([item[0] for item in located], dtype=np.float64)
    warnings: list[str] = []
    if coordinates.size >= 2:
        differences = np.diff(coordinates)
        positive = differences[differences > 1e-4]
        spacing_z = float(np.median(positive)) if positive.size else fallback_spacing
        maximum_gap = float(np.max(differences))
        if np.any(differences <= 1e-4):
            warnings.append("duplicate slice positions")
        if maximum_gap > max(3.0 * spacing_z, 5.0):
            warnings.append(f"gap of {maximum_gap:.3f} mm")
    else:
        spacing_z = fallback_spacing
        maximum_gap = 0.0
    if missing_positions:
        warnings.append("slice positions missing; ordinal order used")
    ordered = [(path, dataset) for _, path, dataset in located]
    origin_lps = (
        np.zeros(3, dtype=np.float64)
        if missing_positions
        else np.asarray(ordered[0][1].ImagePositionPatient, dtype=np.float64)
    )
    return (
        ordered,
        row_direction,
        column_direction,
        origin_lps,
        spacing_z,
        maximum_gap,
        float(pixel_spacing[0]),
        float(pixel_spacing[1]),
        tuple(warnings),
    )


def inspect_patient(
    patient_dir: Path,
    *,
    patient_id: str | None = None,
) -> SeriesInspection:
    patient_id = patient_dir.name if patient_id is None else patient_id
    uid: str | None = None
    headers: list[tuple[Path, Any]] = []
    selected: list[tuple[Path, Any]] = []
    instance_numbers: tuple[int, ...] = ()
    sop_uids: tuple[str, ...] = ()
    try:
        uid, headers = _largest_ct_series(patient_dir)
        selected, instance_numbers, sop_uids = _select_instance_range(headers, None)
        (
            ordered,
            _,
            _,
            _,
            spacing_z,
            maximum_gap,
            row_spacing,
            column_spacing,
            slice_warnings,
        ) = _geometry(selected)
        first = ordered[0][1]
        rows = getattr(first, "Rows", None)
        columns = getattr(first, "Columns", None)
        shape = (
            (int(rows), int(columns), len(ordered))
            if rows is not None and columns is not None
            else None
        )
        return SeriesInspection(
            patient_id=patient_id,
            patient_dir=patient_dir,
            series_uid=uid,
            file_count=len(ordered),
            shape=shape,
            spacing_mm=(row_spacing, column_spacing, spacing_z),
            median_slice_spacing_mm=spacing_z,
            maximum_slice_gap_mm=maximum_gap,
            geometry_status="eligible",
            reason="; ".join(slice_warnings) or None,
            selected_instance_numbers=instance_numbers,
            selected_sop_instance_uids=sop_uids,
        )
    except DicomGeometryError as exc:
        if not headers:
            groups = _read_headers(patient_dir)
            if groups:
                uid, headers = max(groups.items(), key=lambda item: (len(item[1]), item[0]))
        selected_count = len(selected) if selected else len(headers)
        status = "skipped_geometry_gap" if "discontinuous stack" in str(exc) else "invalid_geometry"
        return SeriesInspection(
            patient_id=patient_id,
            patient_dir=patient_dir,
            series_uid=uid,
            file_count=selected_count,
            shape=None,
            spacing_mm=None,
            median_slice_spacing_mm=None,
            maximum_slice_gap_mm=None,
            geometry_status=status,
            reason=str(exc),
            selected_instance_numbers=instance_numbers,
            selected_sop_instance_uids=sop_uids,
        )


def load_dicom_volume(
    patient_dir: Path,
    *,
    patient_id: str | None = None,
) -> DicomVolume:
    uid, headers = _largest_ct_series(patient_dir)
    selected, instance_numbers, sop_uids = _select_instance_range(headers, None)
    (
        ordered,
        row_direction,
        column_direction,
        origin_lps,
        spacing_z,
        maximum_gap,
        row_spacing,
        column_spacing,
        _,
    ) = _geometry(selected)
    slices: list[np.ndarray] = []
    for path, _ in ordered:
        dataset = pydicom.dcmread(path, force=True)
        pixels = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        slices.append(pixels * slope + intercept)
    volume = np.stack(slices, axis=2).astype(np.float32, copy=False)
    slice_normal = np.cross(column_direction, row_direction)
    slice_normal /= np.linalg.norm(slice_normal)
    affine_lps = np.eye(4, dtype=np.float64)
    affine_lps[:3, 0] = row_direction * row_spacing
    affine_lps[:3, 1] = column_direction * column_spacing
    affine_lps[:3, 2] = slice_normal * spacing_z
    affine_lps[:3, 3] = origin_lps
    lps_to_ras = np.diag([-1.0, -1.0, 1.0, 1.0])
    affine_ras = lps_to_ras @ affine_lps
    return DicomVolume(
        patient_id=patient_dir.name if patient_id is None else patient_id,
        series_uid=uid,
        volume_hu=volume,
        affine_ras=affine_ras,
        spacing_mm=(row_spacing, column_spacing, spacing_z),
        files=[path for path, _ in ordered],
        median_slice_spacing_mm=spacing_z,
        maximum_slice_gap_mm=maximum_gap,
        selected_instance_numbers=instance_numbers,
        selected_sop_instance_uids=sop_uids,
    )


def series_sha256(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()
