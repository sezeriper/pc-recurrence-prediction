from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pc_recurrence.image_roi import pipeline
from pc_recurrence.image_roi.dicom import DicomVolume, SeriesInspection
from pc_recurrence.image_roi.model import RuntimeInfo, SegmentationResult


def _inspection(patient_dir: Path) -> SeriesInspection:
    return SeriesInspection(
        patient_id="Patient 1",
        patient_dir=patient_dir,
        series_uid="series",
        file_count=2,
        shape=(8, 8, 8),
        spacing_mm=(1.0, 1.0, 1.0),
        median_slice_spacing_mm=1.0,
        maximum_slice_gap_mm=1.0,
        geometry_status="eligible",
        reason=None,
    )


def test_negative_case_creates_no_patient_artifacts_and_resumes(
    tmp_path: Path, monkeypatch
) -> None:
    patient_dir = tmp_path / "dicom" / "Patient 1"
    patient_dir.mkdir(parents=True)
    inspection = _inspection(patient_dir)
    model_path = tmp_path / "model.ts"
    model_path.touch()
    runtime = RuntimeInfo(
        torch_version="test",
        hip_version="test",
        device_index=0,
        device_name="test",
        total_gpu_bytes=1,
        free_gpu_bytes=1,
        smoke_test_seconds=0.1,
        smoke_test_shape=(4, 3, 96, 96, 96),
    )
    volume = DicomVolume(
        patient_id="Patient 1",
        series_uid="series",
        volume_hu=np.zeros((8, 8, 8), dtype=np.float32),
        affine_ras=np.eye(4),
        spacing_mm=(1.0, 1.0, 1.0),
        files=[],
        median_slice_spacing_mm=1.0,
        maximum_slice_gap_mm=1.0,
    )
    segmentation = SegmentationResult(
        labels_original=np.zeros((8, 8, 8), dtype=np.uint8),
        preprocessed_shape=(8, 8, 8),
        patch_count=1,
        inference_seconds=0.1,
    )
    monkeypatch.setattr(pipeline, "inspect_dataset", lambda *_args, **_kwargs: [inspection])
    monkeypatch.setattr(pipeline, "acquire_model", lambda *_args, **_kwargs: model_path)
    monkeypatch.setattr(pipeline, "validate_and_load_runtime", lambda _path: (object(), runtime))
    monkeypatch.setattr(pipeline, "load_dicom_volume", lambda *_args, **_kwargs: volume)
    monkeypatch.setattr(pipeline, "series_sha256", lambda _files: "checksum")
    monkeypatch.setattr(pipeline, "segment_volume", lambda *_args, **_kwargs: segmentation)

    run_dir = tmp_path / "run"
    pipeline.run_segmentation(
        tmp_path / "dicom",
        tmp_path / "outputs",
        tmp_path / "models",
        run_dir=run_dir,
    )

    assert not (run_dir / "Patient 1").exists()
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["status_counts"] == {"no_pancreas": 1}

    monkeypatch.setattr(
        pipeline,
        "segment_volume",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("resume recomputed")),
    )
    pipeline.run_segmentation(
        tmp_path / "dicom",
        tmp_path / "outputs",
        tmp_path / "models",
        run_dir=run_dir,
        resume=True,
    )
    assert not (run_dir / "Patient 1").exists()


def test_pancreas_target_creates_roi_without_tumor(tmp_path: Path, monkeypatch) -> None:
    patient_dir = tmp_path / "dicom" / "Patient 1"
    patient_dir.mkdir(parents=True)
    inspection = _inspection(patient_dir)
    model_path = tmp_path / "model.ts"
    model_path.touch()
    runtime = RuntimeInfo(
        torch_version="test",
        hip_version="test",
        device_index=0,
        device_name="test",
        total_gpu_bytes=1,
        free_gpu_bytes=1,
        smoke_test_seconds=0.1,
        smoke_test_shape=(4, 3, 96, 96, 96),
    )
    volume = DicomVolume(
        patient_id="Patient 1",
        series_uid="series",
        volume_hu=np.zeros((8, 8, 8), dtype=np.float32),
        affine_ras=np.eye(4),
        spacing_mm=(1.0, 1.0, 1.0),
        files=[],
        median_slice_spacing_mm=1.0,
        maximum_slice_gap_mm=1.0,
    )
    labels = np.zeros((8, 8, 8), dtype=np.uint8)
    labels[2:6, 1:7, 3:6] = 1
    segmentation = SegmentationResult(
        labels_original=labels,
        preprocessed_shape=(8, 8, 8),
        patch_count=1,
        inference_seconds=0.1,
    )
    monkeypatch.setattr(pipeline, "inspect_dataset", lambda *_args, **_kwargs: [inspection])
    monkeypatch.setattr(pipeline, "acquire_model", lambda *_args, **_kwargs: model_path)
    monkeypatch.setattr(pipeline, "validate_and_load_runtime", lambda _path: (object(), runtime))
    monkeypatch.setattr(pipeline, "load_dicom_volume", lambda *_args, **_kwargs: volume)
    monkeypatch.setattr(pipeline, "series_sha256", lambda _files: "checksum")
    monkeypatch.setattr(pipeline, "segment_volume", lambda *_args, **_kwargs: segmentation)

    run_dir = tmp_path / "run"
    pipeline.run_segmentation(
        tmp_path / "dicom",
        tmp_path / "outputs",
        tmp_path / "models",
        run_dir=run_dir,
    )

    patient_output = run_dir / "Patient 1"
    assert patient_output.is_dir()
    assert (patient_output / "review_montage.png").is_file()
    bbox = json.loads((patient_output / "bbox.json").read_text(encoding="utf-8"))
    assert bbox["roi_target"] == "pancreas"
    assert bbox["roi_volume_mm3"] == 72.0
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["roi_target"] == "pancreas"
    assert manifest["status_counts"] == {"detected": 1}
