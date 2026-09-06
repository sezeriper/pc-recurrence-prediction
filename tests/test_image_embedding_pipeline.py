from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from pc_recurrence.image_data.dicom import DicomVolume
from pc_recurrence.image_embedding import pipeline
from pc_recurrence.image_embedding.constants import ImageEncoderName
from pc_recurrence.image_embedding.foundation_models import FoundationModelArtifacts, RuntimeInfo


def test_discovery_can_skip_missing_curated_patient_directories(
    tmp_path: Path, monkeypatch
) -> None:
    dicom_root = tmp_path / "dicom_selected"
    dicom_root.mkdir()
    (dicom_root / "curation_manifest.json").write_text("{}", encoding="utf-8")
    row = SimpleNamespace(patient_id="Patient without images", dicom_folder="PATIENT404")
    monkeypatch.setattr(pipeline, "load_image_workbook", lambda _path: [row])
    monkeypatch.setattr(pipeline, "select_image_workbook_rows", lambda rows, _patients: rows)

    discovery = pipeline.discover_ct_series_cases(
        dicom_root,
        tmp_path / "workbook.xlsx",
    )

    assert discovery == []
    assert discovery.skipped == [
        {
            "patient_id": "Patient without images",
            "status": "skipped",
            "reason": "no curated CT series directory",
        }
    ]


def _selected_ct_source(
    tmp_path: Path, monkeypatch
) -> tuple[Path, Path, pipeline.CtSeriesCase]:
    dicom_root = tmp_path / "dicom_selected"
    patient_dir = dicom_root / "PATIENT4"
    patient_dir.mkdir(parents=True)
    (dicom_root / "curation_manifest.json").write_text("{}", encoding="utf-8")
    workbook = tmp_path / "patients.xlsx"
    workbook.touch()
    case = pipeline.CtSeriesCase("Patient 4", patient_dir, "study", "series")
    volume = DicomVolume(
        patient_id="Patient 4",
        study_uid="study",
        series_uid="series",
        volume_hu=np.full((28, 30, 20), -500.0, dtype=np.float32),
        affine_ras=np.diag([2.0, 2.0, 2.5, 1.0]),
        spacing_mm=(2.0, 2.0, 2.5),
        files=[],
        median_slice_spacing_mm=2.5,
        maximum_slice_gap_mm=2.5,
        selected_instance_numbers=tuple(range(20)),
        selected_sop_instance_uids=tuple(f"sop-{index}" for index in range(20)),
    )
    monkeypatch.setattr(pipeline, "discover_ct_series_cases", lambda *_args, **_kwargs: [case])
    monkeypatch.setattr(pipeline, "load_dicom_volume", lambda *_args, **_kwargs: volume)
    monkeypatch.setattr(pipeline, "series_sha256", lambda _files: "C" * 64)
    return dicom_root, workbook, case


def _mock_runtime(
    tmp_path: Path,
    monkeypatch,
    encoder_name: ImageEncoderName,
    dimension: int,
) -> None:
    model_file = tmp_path / f"{encoder_name.value}.pt"
    model_file.touch()
    artifact_count = 2 if encoder_name is ImageEncoderName.SPECTRE else 1
    artifacts = FoundationModelArtifacts(
        paths=tuple(model_file for _ in range(artifact_count)),
        hashes=tuple("A" * 64 for _ in range(artifact_count)),
    )
    runtime = RuntimeInfo(
        torch_version="test",
        monai_version="test",
        device_type="cpu",
        device_name="test",
        total_device_bytes=1,
        free_device_bytes=1,
        smoke_test_seconds=0.1,
        smoke_test_shape=(dimension,),
    )
    monkeypatch.setattr(pipeline, "acquire_foundation_model", lambda *_args, **_kwargs: artifacts)
    monkeypatch.setattr(pipeline, "load_foundation_runtime", lambda *_args: (object(), runtime))
    monkeypatch.setattr(
        pipeline,
        "encode_spectre",
        lambda _model, _data, *, expected_grid: (
            np.ones((int(np.prod(expected_grid)) + 1, 1080), dtype=np.float32),
            expected_grid,
        ),
    )
    monkeypatch.setattr(
        pipeline,
        "encode_merlin",
        lambda _model, _data: np.ones(2048, dtype=np.float32),
    )


@pytest.mark.parametrize(
    ("encoder_name", "dimension"),
    [
        (ImageEncoderName.SPECTRE, 1080),
        (ImageEncoderName.MERLIN, 2048),
    ],
)
def test_foundation_encoders_use_selected_ct_center(
    tmp_path: Path,
    monkeypatch,
    encoder_name: ImageEncoderName,
    dimension: int,
) -> None:
    dicom_root, workbook, _ = _selected_ct_source(tmp_path, monkeypatch)
    _mock_runtime(tmp_path, monkeypatch, encoder_name, dimension)

    output = pipeline.run_embedding(
        dicom_root,
        tmp_path / "outputs",
        tmp_path / "models",
        workbook_path=workbook,
        encoder_name=encoder_name,
        run_dir=tmp_path / f"{encoder_name.value}_run",
    )

    with np.load(output / "image_embeddings.npz", allow_pickle=False) as artifact:
        assert artifact["encoder"].item() == encoder_name.value
        assert artifact["embeddings"].shape == (1, dimension)
        assert np.isfinite(artifact["embeddings"]).all()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["stage"] == f"{encoder_name.value}_selected_ct_embedding"
    assert manifest["source_kind"] == "workbook-selected CT series range"
    assert manifest["preprocessing"]["centering"] == "CT volume center"
    record = manifest["patients"][0]
    assert record["ct_series_sha256"] == "C" * 64
    np.testing.assert_allclose(
        record["preprocessing"]["crop_center_voxel"],
        record["preprocessing"]["volume_center_voxel"],
    )


def test_selected_ct_cache_reuses_matching_series(tmp_path: Path, monkeypatch) -> None:
    dicom_root, workbook, _ = _selected_ct_source(tmp_path, monkeypatch)
    _mock_runtime(tmp_path, monkeypatch, ImageEncoderName.SPECTRE, 1080)
    encode_calls: list[str] = []

    def _fake_encode_spectre(model, data, *, expected_grid):
        encode_calls.append("encoded")
        return (
            np.ones((int(np.prod(expected_grid)) + 1, 1080), dtype=np.float32),
            expected_grid,
        )

    monkeypatch.setattr(pipeline, "encode_spectre", _fake_encode_spectre)
    run_dir = tmp_path / "embedding_run"
    for _ in range(2):
        pipeline.run_embedding(
            dicom_root,
            tmp_path / "outputs",
            tmp_path / "models",
            workbook_path=workbook,
            encoder_name=ImageEncoderName.SPECTRE,
            run_dir=run_dir,
        )

    assert encode_calls == ["encoded"]
    state = json.loads(next((run_dir / ".state").glob("*.json")).read_text(encoding="utf-8"))
    assert state["ct_series_sha256"] == "C" * 64
