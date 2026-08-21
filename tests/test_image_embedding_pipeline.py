from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from pc_recurrence.image_embedding import pipeline
from pc_recurrence.image_embedding.constants import CenteringMode, ImageEncoderName
from pc_recurrence.image_embedding.foundation_models import FoundationModelArtifacts, RuntimeInfo


@pytest.mark.parametrize(
    ("encoder_name", "dimension"),
    [
        (ImageEncoderName.SPECTRE, 1080),
        (ImageEncoderName.MERLIN, 2048),
    ],
)
def test_foundation_encoders_write_finite_dynamic_embeddings(
    tmp_path: Path,
    monkeypatch,
    encoder_name: ImageEncoderName,
    dimension: int,
) -> None:
    roi_run = tmp_path / "roi_run"
    patient_dir = roi_run / "Patient 4"
    patient_dir.mkdir(parents=True)
    (roi_run / "run_manifest.json").write_text(
        json.dumps({"roi_target": "pancreas"}), encoding="utf-8"
    )
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    ct = np.full((20, 22, 18), -1000.0, dtype=np.float32)
    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[7:12, 3:9, 5:11] = 1
    nib.save(nib.Nifti1Image(ct, affine), patient_dir / "roi_ct.nii.gz")
    nib.save(
        nib.Nifti1Image(mask, affine),
        patient_dir / "roi_pancreas_mask.nii.gz",
    )
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
        hip_version="test",
        device_index=0,
        device_name="test",
        total_gpu_bytes=1,
        free_gpu_bytes=1,
        smoke_test_seconds=0.1,
        smoke_test_shape=(dimension,),
    )
    monkeypatch.setattr(
        pipeline, "acquire_foundation_model", lambda *_args, **_kwargs: artifacts
    )
    monkeypatch.setattr(
        pipeline, "load_foundation_runtime", lambda *_args: (object(), runtime)
    )
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

    output = pipeline.run_embedding(
        roi_run,
        tmp_path / "outputs",
        tmp_path / "models",
        encoder_name=encoder_name,
        run_dir=tmp_path / f"{encoder_name.value}_run",
    )

    with np.load(output / "image_embeddings.npz", allow_pickle=False) as artifact:
        assert artifact["encoder"].item() == encoder_name.value
        assert artifact["embeddings"].shape == (1, dimension)
        assert np.isfinite(artifact["embeddings"]).all()
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["encoder"] == encoder_name.value
    assert manifest["feature_extraction"]["embedding_dimension"] == dimension


def test_volume_centering_invalidates_cache_and_records_mode(
    tmp_path: Path, monkeypatch
) -> None:
    roi_run = tmp_path / "roi_run"
    patient_dir = roi_run / "Patient 4"
    patient_dir.mkdir(parents=True)
    (roi_run / "run_manifest.json").write_text(
        json.dumps({"roi_target": "pancreas"}), encoding="utf-8"
    )
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    ct = np.full((28, 30, 20), -1000.0, dtype=np.float32)
    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[2:6, 22:27, 3:7] = 1
    nib.save(nib.Nifti1Image(ct, affine), patient_dir / "roi_ct.nii.gz")
    nib.save(
        nib.Nifti1Image(mask, affine),
        patient_dir / "roi_pancreas_mask.nii.gz",
    )
    model_file = tmp_path / "spectre.pt"
    model_file.touch()
    artifacts = FoundationModelArtifacts(
        paths=(model_file, model_file), hashes=("A" * 64, "B" * 64)
    )
    runtime = RuntimeInfo("test", "test", "test", 0, "test", 1, 1, 0.1, (2, 1080))
    monkeypatch.setattr(
        pipeline, "acquire_foundation_model", lambda *_args, **_kwargs: artifacts
    )
    monkeypatch.setattr(
        pipeline, "load_foundation_runtime", lambda *_args: (object(), runtime)
    )
    encode_calls: list[str] = []

    def _fake_encode_spectre(model, data, *, expected_grid):
        encode_calls.append("encoded")
        return (
            np.ones((int(np.prod(expected_grid)) + 1, 1080), dtype=np.float32),
            expected_grid,
        )

    monkeypatch.setattr(pipeline, "encode_spectre", _fake_encode_spectre)

    run_dir = tmp_path / "embedding_run"
    first = pipeline.run_embedding(
        roi_run,
        tmp_path / "outputs",
        tmp_path / "models",
        encoder_name=ImageEncoderName.SPECTRE,
        run_dir=run_dir,
        centering=CenteringMode.PANCREAS,
    )
    first_manifest = json.loads((first / "run_manifest.json").read_text(encoding="utf-8"))
    assert first_manifest["preprocessing"]["centering"] == "predicted pancreas bounding-box center"

    second = pipeline.run_embedding(
        roi_run,
        tmp_path / "outputs",
        tmp_path / "models",
        encoder_name=ImageEncoderName.SPECTRE,
        run_dir=run_dir,
        centering=CenteringMode.VOLUME,
    )

    assert len(encode_calls) == 2  # volume run recomputed despite cached pancreas state
    second_manifest = json.loads((second / "run_manifest.json").read_text(encoding="utf-8"))
    assert second_manifest["preprocessing"]["centering"] == "CT volume center"
    state = json.loads(
        next((second / ".state").glob("*.json")).read_text(encoding="utf-8")
    )
    assert state["centering"] == "volume"
    assert state["record"]["preprocessing"]["centering"] == "volume"
    np.testing.assert_allclose(
        state["record"]["preprocessing"]["crop_center_voxel"],
        state["record"]["preprocessing"]["volume_center_voxel"],
    )


def test_foundation_encoder_strictly_skips_missing_pancreas_mask(
    tmp_path: Path, monkeypatch
) -> None:
    roi_run = tmp_path / "roi_run"
    patient_dir = roi_run / "Patient 4"
    patient_dir.mkdir(parents=True)
    (roi_run / "run_manifest.json").write_text(
        json.dumps({"roi_target": "pancreas"}), encoding="utf-8"
    )
    nib.save(
        nib.Nifti1Image(np.zeros((8, 8, 8), dtype=np.float32), np.eye(4)),
        patient_dir / "roi_ct.nii.gz",
    )
    model_file = tmp_path / "spectre.pt"
    model_file.touch()
    artifacts = FoundationModelArtifacts(
        paths=(model_file, model_file), hashes=("A" * 64, "B" * 64)
    )
    runtime = RuntimeInfo("test", "test", "test", 0, "test", 1, 1, 0.1, (2, 1080))
    monkeypatch.setattr(
        pipeline, "acquire_foundation_model", lambda *_args, **_kwargs: artifacts
    )
    monkeypatch.setattr(
        pipeline, "load_foundation_runtime", lambda *_args: (object(), runtime)
    )

    output = pipeline.run_embedding(
        roi_run,
        tmp_path / "outputs",
        tmp_path / "models",
        encoder_name=ImageEncoderName.SPECTRE,
        run_dir=tmp_path / "spectre_run",
    )

    with np.load(output / "image_embeddings.npz", allow_pickle=False) as artifact:
        assert artifact["embeddings"].shape == (0, 1080)
    manifest = json.loads((output / "run_manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["failures"]) == 1
    assert "roi_pancreas_mask" in manifest["failures"][0]["reason"]
