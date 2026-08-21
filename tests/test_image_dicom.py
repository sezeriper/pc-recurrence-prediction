from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian, generate_uid

from pc_recurrence.image_roi.dicom import (
    DicomGeometryError,
    inspect_patient,
    load_dicom_volume,
    natural_patient_key,
    parse_image_range,
    select_series_files,
)
from pc_recurrence.image_roi.pipeline import inspect_dataset


def _write_slice(
    path: Path,
    *,
    series_uid: str,
    z: float,
    stored_value: int,
    slope: float = 2.0,
    intercept: float = -1000.0,
    instance_number: int | None = None,
) -> None:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = CTImageStorage
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.SeriesInstanceUID = series_uid
    dataset.StudyInstanceUID = generate_uid()
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
    dataset.PixelData = np.full((4, 5), stored_value, dtype=np.int16).tobytes()
    dataset.save_as(path, enforce_file_format=True)


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

    volume = load_dicom_volume(patient)

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
    for index, z in enumerate([0.0, 0.5, 1.0, 20.0]):
        _write_slice(patient / f"{index}.dcm", series_uid=uid, z=z, stored_value=index)

    inspection = inspect_patient(patient)

    assert inspection.geometry_status == "eligible"
    assert "gap" in (inspection.reason or "")
    assert inspection.maximum_slice_gap_mm == pytest.approx(19.0)
    assert inspection.file_count == 4


def test_range_selection_maps_instance_ordinals_and_folder_loads_all(tmp_path: Path) -> None:
    patient = tmp_path / "Patient ranged"
    patient.mkdir()
    uid = generate_uid()
    _write_slice(
        patient / "one.dcm",
        series_uid=uid,
        z=2.0,
        stored_value=30,
        instance_number=1,
    )
    _write_slice(
        patient / "two.dcm",
        series_uid=uid,
        z=1.0,
        stored_value=20,
        instance_number=2,
    )
    _write_slice(
        patient / "three.dcm",
        series_uid=uid,
        z=0.0,
        stored_value=10,
        instance_number=3,
    )

    assert parse_image_range(" 1 - 2 ") == (1, 2)
    files, instance_numbers, sop_uids = select_series_files(patient, "1-2")

    assert [path.name for path in files] == ["one.dcm", "two.dcm"]
    assert instance_numbers == (1, 2)
    assert len(sop_uids) == 2

    inspection = inspect_patient(patient)
    volume = load_dicom_volume(patient)
    assert inspection.geometry_status == "eligible"
    assert inspection.file_count == 3
    assert sorted(inspection.selected_instance_numbers) == [1, 2, 3]
    assert len(inspection.selected_sop_instance_uids) == 3
    assert volume.volume_hu[0, 0].tolist() == pytest.approx([-980, -960, -940])


def test_invalid_image_range_is_strictly_excluded(tmp_path: Path) -> None:
    patient = tmp_path / "Patient invalid range"
    patient.mkdir()
    uid = generate_uid()
    for instance, z in enumerate([0.0, 1.0, 2.0], start=1):
        _write_slice(
            patient / f"{instance}.dcm",
            series_uid=uid,
            z=z,
            stored_value=instance,
            instance_number=instance,
        )

    with pytest.raises(DicomGeometryError, match="exceeds 3 series slices"):
        select_series_files(patient, "2-5")


def test_discontinuous_selected_range_retains_sop_uid_audit(tmp_path: Path) -> None:
    patient = tmp_path / "Patient discontinuous range"
    patient.mkdir()
    uid = generate_uid()
    for instance, z in enumerate([0.0, 1.0, 2.0, 20.0], start=1):
        _write_slice(
            patient / f"{instance}.dcm",
            series_uid=uid,
            z=z,
            stored_value=instance,
            instance_number=instance,
        )

    files, instance_numbers, sop_uids = select_series_files(patient, "1-4")
    assert len(files) == 4
    assert instance_numbers == (1, 2, 3, 4)
    assert len(sop_uids) == 4

    inspection = inspect_patient(patient)
    assert inspection.geometry_status == "eligible"
    assert "gap" in (inspection.reason or "")
    assert inspection.file_count == 4
    assert inspection.selected_instance_numbers == (1, 2, 3, 4)
    assert len(inspection.selected_sop_instance_uids) == 4


def test_real_curated_dataset_has_expected_geometry_statuses() -> None:
    root = Path("dataset/dicom_selected")
    workbook = Path("dataset/pankreas adeno ca 10 hasta.xlsx")
    if not root.exists() or not workbook.exists():
        pytest.skip("local research dataset is unavailable")
    inspections = inspect_dataset(root, workbook_path=workbook)
    assert len(inspections) == 9
    statuses = {item.patient_id: item.geometry_status for item in inspections}
    assert all(
        statuses[f"Patient {index}"] == "eligible" for index in range(2, 11)
    )
    patient_10 = next(item for item in inspections if item.patient_id == "Patient 10")
    assert "gap" in (patient_10.reason or "")
    assert patient_10.maximum_slice_gap_mm == pytest.approx(64.401, abs=0.01)
    assert patient_10.file_count == 278
    assert {item.patient_dir.name for item in inspections} == {
        "PATIENT2321275",
        "PATIENT2481647",
        "PATIENT2598080",
        "PATIENT2625090",
        "PATIENT2647442",
        "PATIENT3110212",
        "PATIENT3940389",
        "PATIENT4201780",
        "PATIENT853534",
    }
