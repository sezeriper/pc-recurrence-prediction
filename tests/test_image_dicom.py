from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from pc_recurrence.image_data.constants import DEFAULT_DICOM_ROOT
from pc_recurrence.image_data.dicom import (
    DicomGeometryError,
    SeriesKey,
    discover_dicom_series,
    inspect_patient,
    load_dicom_volume,
    natural_patient_key,
    parse_image_range,
    select_series_files,
)
from pc_recurrence.image_data.inspection import inspect_dataset

DEFAULT_STUDY_UID = "1.2.826.0.1.3680043.10.1000"


def _write_slice(
    path: Path,
    *,
    series_uid: str,
    z: float,
    stored_value: int,
    study_uid: str = DEFAULT_STUDY_UID,
    sop_uid: str | None = None,
    sop_class: str = str(CTImageStorage),
    slope: float = 2.0,
    intercept: float = -1000.0,
    instance_number: int | None = None,
    metadata: dict[str, Any] | None = None,
) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    sop_uid = sop_uid or generate_uid()
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = sop_class
    meta.MediaStorageSOPInstanceUID = sop_uid
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = sop_class
    dataset.SOPInstanceUID = sop_uid
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyInstanceUID = study_uid
    dataset.Modality = "CT"
    dataset.Rows = 4
    dataset.Columns = 5
    dataset.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    dataset.ImagePositionPatient = [10, 20, z]
    dataset.PixelSpacing = [2, 3]
    dataset.SliceThickness = 9
    if instance_number is not None:
        dataset.InstanceNumber = instance_number
    dataset.RescaleSlope = slope
    dataset.RescaleIntercept = intercept
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 16
    dataset.HighBit = 15
    dataset.PixelRepresentation = 1
    for name, value in (metadata or {}).items():
        setattr(dataset, name, value)
    dataset.PixelData = np.full((4, 5), stored_value, dtype=np.int16).tobytes()
    dataset.save_as(path, enforce_file_format=True)
    return sop_uid


def _key(series_uid: str, study_uid: str = DEFAULT_STUDY_UID) -> SeriesKey:
    return SeriesKey(study_uid, series_uid)


def test_natural_patient_order() -> None:
    values = [Path("Patient 10"), Path("Patient 2"), Path("Patient 1")]
    assert [path.name for path in sorted(values, key=natural_patient_key)] == [
        "Patient 1",
        "Patient 2",
        "Patient 10",
    ]


def test_load_sorts_spatially_uses_position_spacing_and_applies_hu(tmp_path: Path) -> None:
    patient = tmp_path / "Patient 1"
    patient.mkdir()
    uid = generate_uid()
    _write_slice(patient / "c.dcm", series_uid=uid, z=1.0, stored_value=30)
    _write_slice(patient / "a.dcm", series_uid=uid, z=0.0, stored_value=10)
    _write_slice(patient / "b.dcm", series_uid=uid, z=0.5, stored_value=20)

    volume = load_dicom_volume(patient, selection=_key(uid))

    assert volume.study_uid == DEFAULT_STUDY_UID
    assert volume.series_uid == uid
    assert volume.shape == (4, 5, 3)
    assert volume.spacing_mm == pytest.approx((2.0, 3.0, 0.5))
    assert volume.volume_hu[0, 0].tolist() == pytest.approx([-980, -960, -940])
    assert np.linalg.norm(volume.affine_ras[:3, 0]) == pytest.approx(2.0)
    assert np.linalg.norm(volume.affine_ras[:3, 1]) == pytest.approx(3.0)
    assert np.linalg.norm(volume.affine_ras[:3, 2]) == pytest.approx(0.5)
    assert volume.affine_ras[:3, 3].tolist() == pytest.approx([-10, -20, 0])


def test_discontinuous_stack_is_accepted_with_warning(tmp_path: Path) -> None:
    patient = tmp_path / "Patient 2"
    patient.mkdir()
    uid = generate_uid()
    for index, z in enumerate([0.0, 0.5, 1.0, 20.0], start=1):
        _write_slice(
            patient / f"{index}.dcm",
            series_uid=uid,
            z=z,
            stored_value=index,
            instance_number=index,
        )
    inspection = inspect_patient(patient, selection=_key(uid))
    assert inspection.geometry_status == "eligible"
    assert "gap" in (inspection.reason or "")
    assert inspection.maximum_slice_gap_mm == pytest.approx(19.0)


def test_range_selection_is_exact_and_uses_instance_ordinals(tmp_path: Path) -> None:
    patient = tmp_path / "Patient ranged"
    patient.mkdir()
    selected_uid = generate_uid()
    larger_uid = generate_uid()
    for instance, z in enumerate([2.0, 1.0, 0.0], start=1):
        _write_slice(
            patient / f"chosen-{instance}.dcm",
            series_uid=selected_uid,
            z=z,
            stored_value=instance * 10,
            instance_number=instance,
        )
    for instance in range(1, 6):
        _write_slice(
            patient / f"large-{instance}.dcm",
            series_uid=larger_uid,
            z=float(instance),
            stored_value=99,
            instance_number=instance,
        )

    assert parse_image_range(" 0 - 1 ") == (0, 1)
    files, instance_numbers, sop_uids = select_series_files(patient, _key(selected_uid), "0-1")
    assert [path.name for path in files] == ["chosen-1.dcm", "chosen-2.dcm"]
    assert instance_numbers == (1, 2)
    assert len(sop_uids) == 2
    with pytest.raises(DicomGeometryError, match="expected exactly one"):
        load_dicom_volume(patient)
    volume = load_dicom_volume(patient, selection=_key(selected_uid))
    assert volume.shape == (4, 5, 3)


def test_invalid_and_missing_exact_selection_fail(tmp_path: Path) -> None:
    patient = tmp_path / "Patient invalid range"
    patient.mkdir()
    uid = generate_uid()
    for instance in range(1, 4):
        _write_slice(
            patient / f"{instance}.dcm",
            series_uid=uid,
            z=float(instance),
            stored_value=instance,
            instance_number=instance,
        )
    with pytest.raises(DicomGeometryError, match="exceeds 3 series slices"):
        select_series_files(patient, _key(uid), "1-3")
    with pytest.raises(DicomGeometryError, match="not found"):
        select_series_files(patient, _key(generate_uid()), "1-2")


def test_identical_sop_copies_collapse_and_preserve_sources(tmp_path: Path) -> None:
    patient = tmp_path / "Patient duplicate"
    uid = generate_uid()
    original = patient / "ST1" / "SE1" / "one.dcm"
    _write_slice(original, series_uid=uid, z=0.0, stored_value=1, instance_number=1)
    copied = patient / "copy" / "one-copy.dcm"
    copied.parent.mkdir(parents=True)
    shutil.copy2(original, copied)
    _write_slice(
        patient / "ST1" / "SE1" / "two.dcm",
        series_uid=uid,
        z=1.0,
        stored_value=2,
        instance_number=2,
    )

    series = discover_dicom_series(patient).series[0]
    assert series.source_file_count == 3
    assert len(series.headers) == 2
    assert series.duplicate_file_count == 1
    assert series.source_directories == ("copy", "ST1/SE1")
    files, _, _ = select_series_files(patient, _key(uid))
    assert original in files
    assert copied not in files


def test_conflicting_sop_copies_reject_candidate(tmp_path: Path) -> None:
    patient = tmp_path / "Patient conflict"
    uid = generate_uid()
    sop_uid = generate_uid()
    _write_slice(
        patient / "a.dcm",
        series_uid=uid,
        sop_uid=sop_uid,
        z=0,
        stored_value=1,
        instance_number=1,
    )
    _write_slice(
        patient / "b.dcm",
        series_uid=uid,
        sop_uid=sop_uid,
        z=0,
        stored_value=2,
        instance_number=1,
    )
    series = discover_dicom_series(patient).series[0]
    assert "conflicting file bytes" in series.problems[0]
    with pytest.raises(DicomGeometryError, match="conflicting file bytes"):
        select_series_files(patient, _key(uid))


def test_series_order_uses_number_then_directory_and_identity(tmp_path: Path) -> None:
    patient = tmp_path / "Patient ordered"
    one = generate_uid()
    two = generate_uid()
    missing = generate_uid()
    for uid, folder, number in ((two, "z", 2), (missing, "a", None), (one, "b", 1)):
        for instance in (1, 2):
            metadata = {} if number is None else {"SeriesNumber": number}
            _write_slice(
                patient / folder / f"{instance}.dcm",
                series_uid=uid,
                z=float(instance),
                stored_value=instance,
                instance_number=instance,
                metadata=metadata,
            )
    assert [item.key.series_uid for item in discover_dicom_series(patient).series] == [
        one,
        two,
        missing,
    ]


def test_same_series_uid_in_two_studies_stays_distinct(tmp_path: Path) -> None:
    patient = tmp_path / "Patient studies"
    series_uid = generate_uid()
    studies = (generate_uid(), generate_uid())
    for study_index, study_uid in enumerate(studies):
        for instance in (1, 2):
            _write_slice(
                patient / str(study_index) / f"{instance}.dcm",
                study_uid=study_uid,
                series_uid=series_uid,
                z=float(instance),
                stored_value=instance,
                instance_number=instance,
            )
    discovery = discover_dicom_series(patient)
    assert {item.key for item in discovery.series} == {
        SeriesKey(studies[0], series_uid),
        SeriesKey(studies[1], series_uid),
    }


def test_no_selector_requires_exactly_one_processable_series(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DicomGeometryError, match="no processable"):
        load_dicom_volume(empty)
    patient = tmp_path / "single"
    patient.mkdir()
    uid = generate_uid()
    for instance in (1, 2):
        _write_slice(
            patient / f"{instance}.dcm",
            series_uid=uid,
            z=float(instance),
            stored_value=instance,
            instance_number=instance,
        )
    assert load_dicom_volume(patient).series_uid == uid


def test_real_curated_dataset_matches_explicit_manifest() -> None:
    manifest_path = DEFAULT_DICOM_ROOT / "curation_manifest.json"
    if not manifest_path.is_file():
        pytest.skip("complete explicit curation manifest is unavailable")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "complete" or not manifest.get("patients"):
        pytest.skip("complete explicit curation manifest is unavailable")
    inspections = inspect_dataset(DEFAULT_DICOM_ROOT)
    by_folder = {item.patient_dir.name: item for item in inspections}
    for patient in manifest["patients"]:
        inspection = by_folder[patient["dicom_folder"]]
        assert inspection.study_uid == patient["study_uid"]
        assert inspection.series_uid == patient["series_uid"]
        assert len(discover_dicom_series(inspection.patient_dir).series) == 1
