from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from pc_recurrence.io import sha256_file
from pc_recurrence.runtime import device_memory_info, inference_autocast, select_device

from .constants import (
    MERLIN_EMBEDDING_DIMENSION,
    MERLIN_FILENAME,
    MERLIN_INPUT_SIZE,
    MERLIN_REPOSITORY,
    MERLIN_REVISION,
    MERLIN_SHA256,
    SPECTRE_BACKBONE_FILENAME,
    SPECTRE_BACKBONE_SHA256,
    SPECTRE_COMBINER_FILENAME,
    SPECTRE_COMBINER_SHA256,
    SPECTRE_CROP_SIZE,
    SPECTRE_EMBEDDING_DIMENSION,
    SPECTRE_REPOSITORY,
    SPECTRE_REVISION,
    ImageEncoderName,
)


class RuntimeValidationError(RuntimeError):
    """Raised when a model artifact or the selected PyTorch runtime is unusable."""


@dataclass(frozen=True)
class RuntimeInfo:
    torch_version: str
    monai_version: str
    device_type: str
    device_name: str
    total_device_bytes: int | None
    free_device_bytes: int | None
    smoke_test_seconds: float
    smoke_test_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "smoke_test_shape": list(self.smoke_test_shape),
        }


@dataclass(frozen=True)
class FoundationModelArtifacts:
    paths: tuple[Path, ...]
    hashes: tuple[str, ...]


def _download_verified(
    *,
    repository: str,
    revision: str,
    filename: str,
    expected_sha256: str,
    local_files_only: bool,
) -> Path:
    path = Path(
        hf_hub_download(
            repo_id=repository,
            revision=revision,
            filename=filename,
            local_files_only=local_files_only,
        )
    )
    actual = sha256_file(path)
    if actual.casefold() != expected_sha256.casefold():
        raise RuntimeValidationError(
            f"model checksum mismatch for {filename}: expected {expected_sha256}, received {actual}"
        )
    return path


def acquire_foundation_model(
    encoder_name: ImageEncoderName,
    cache_dir: Path,
    *,
    local_files_only: bool = False,
) -> FoundationModelArtifacts:
    cache_dir.mkdir(parents=True, exist_ok=True)
    if encoder_name is ImageEncoderName.SPECTRE:
        backbone = _download_verified(
            repository=SPECTRE_REPOSITORY,
            revision=SPECTRE_REVISION,
            filename=SPECTRE_BACKBONE_FILENAME,
            expected_sha256=SPECTRE_BACKBONE_SHA256,
            local_files_only=local_files_only,
        )
        combiner = _download_verified(
            repository=SPECTRE_REPOSITORY,
            revision=SPECTRE_REVISION,
            filename=SPECTRE_COMBINER_FILENAME,
            expected_sha256=SPECTRE_COMBINER_SHA256,
            local_files_only=local_files_only,
        )
        return FoundationModelArtifacts(
            paths=(backbone, combiner),
            hashes=(SPECTRE_BACKBONE_SHA256, SPECTRE_COMBINER_SHA256),
        )
    if encoder_name is ImageEncoderName.MERLIN:
        checkpoint = _download_verified(
            repository=MERLIN_REPOSITORY,
            revision=MERLIN_REVISION,
            filename=MERLIN_FILENAME,
            expected_sha256=MERLIN_SHA256,
            local_files_only=local_files_only,
        )
        return FoundationModelArtifacts(paths=(checkpoint,), hashes=(MERLIN_SHA256,))
    raise ValueError(f"unsupported foundation encoder: {encoder_name}")


def _runtime_info(
    device: torch.device,
    device_name: str,
    started: float,
    smoke_shape: tuple[int, ...],
) -> RuntimeInfo:
    import monai

    free_bytes, total_bytes = device_memory_info(device)
    return RuntimeInfo(
        torch_version=torch.__version__,
        monai_version=monai.__version__,
        device_type=device.type,
        device_name=device_name,
        total_device_bytes=total_bytes,
        free_device_bytes=free_bytes,
        smoke_test_seconds=time.perf_counter() - started,
        smoke_test_shape=smoke_shape,
    )


def _load_spectre(artifacts: FoundationModelArtifacts) -> torch.nn.Module:
    from spectre.model import SpectreImageFeatureExtractor
    from spectre.presets import get_preset

    preset = get_preset("spectre-large")
    return SpectreImageFeatureExtractor(
        backbone_name=preset.backbone,
        backbone_kwargs=preset.backbone_kwargs,
        backbone_checkpoint_path_or_url=str(artifacts.paths[0]),
        feature_combiner_name=preset.feature_combiner,
        feature_combiner_kwargs=preset.feature_combiner_kwargs,
        feature_combiner_checkpoint_path_or_url=str(artifacts.paths[1]),
    )


def _load_merlin(artifacts: FoundationModelArtifacts) -> torch.nn.Module:
    import torchvision
    from merlin.models.i3res import I3ResNet

    resnet = torchvision.models.resnet152(weights=None)
    model = I3ResNet(
        resnet,
        class_nb=1692,
        conv_class=True,
        ImageEmbedding=True,
    )
    checkpoint = torch.load(artifacts.paths[0], map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, (dict, OrderedDict)):
        raise RuntimeValidationError("Merlin checkpoint is not a state dictionary")
    prefix = "encode_image.i3_resnet."
    image_state = {
        key.removeprefix(prefix): value
        for key, value in checkpoint.items()
        if key.startswith(prefix)
    }
    if not image_state:
        raise RuntimeValidationError("Merlin checkpoint contains no image-encoder weights")
    incompatible = model.load_state_dict(image_state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeValidationError(f"Merlin checkpoint mismatch: {incompatible}")
    return model


def load_foundation_runtime(
    encoder_name: ImageEncoderName, artifacts: FoundationModelArtifacts
) -> tuple[torch.nn.Module, RuntimeInfo]:
    device, device_name = select_device()
    started = time.perf_counter()
    if encoder_name is ImageEncoderName.SPECTRE:
        model = _load_spectre(artifacts).to(device).eval()
        smoke = torch.full(SPECTRE_CROP_SIZE, -1000.0, dtype=torch.float32)
        features, _ = encode_spectre(model, smoke.numpy(), expected_grid=(1, 1, 1))
        expected = (1 + 1, SPECTRE_EMBEDDING_DIMENSION)
        if features.shape != expected:
            raise RuntimeValidationError(
                f"SPECTRE smoke test failed: expected {expected}, received {features.shape}"
            )
    elif encoder_name is ImageEncoderName.MERLIN:
        model = _load_merlin(artifacts).to(device).eval()
        vector = encode_merlin(model, np.zeros(MERLIN_INPUT_SIZE, dtype=np.float32))
        expected = (MERLIN_EMBEDDING_DIMENSION,)
        if vector.shape != expected:
            raise RuntimeValidationError(
                f"Merlin smoke test failed: expected {expected}, received {vector.shape}"
            )
    else:
        raise ValueError(f"unsupported foundation encoder: {encoder_name}")
    return model, _runtime_info(device, device_name, started, expected)


def encode_spectre(
    model: torch.nn.Module,
    volume_hu: np.ndarray,
    *,
    expected_grid: tuple[int, int, int],
) -> tuple[np.ndarray, tuple[int, int, int]]:
    data = np.asarray(volume_hu, dtype=np.float32)
    if data.ndim != 3 or not np.isfinite(data).all():
        raise ValueError("SPECTRE input must be a finite 3D HU volume")
    device = next(model.parameters()).device
    tensor = torch.from_numpy(np.ascontiguousarray(data))[None].to(device)
    autocast = inference_autocast(device)
    with torch.inference_mode(), autocast:
        features = model.extract(tensor, max_crops_per_forward=1)
    values = features.to(device="cpu", dtype=torch.float32).numpy()
    if values.ndim != 2 or values.shape[1] != SPECTRE_EMBEDDING_DIMENSION:
        raise RuntimeValidationError(f"unexpected SPECTRE output shape: {values.shape}")
    patch_count = int(np.prod(expected_grid))
    if values.shape[0] != patch_count + 1 or not np.isfinite(values).all():
        raise RuntimeValidationError(
            f"unexpected SPECTRE token count for grid {expected_grid}: {values.shape}"
        )
    return values, expected_grid


def encode_merlin(model: torch.nn.Module, normalized_volume: np.ndarray) -> np.ndarray:
    data = np.asarray(normalized_volume, dtype=np.float32)
    if data.shape != MERLIN_INPUT_SIZE or not np.isfinite(data).all():
        raise ValueError(f"Merlin input must have shape {MERLIN_INPUT_SIZE}")
    device = next(model.parameters()).device
    tensor = torch.from_numpy(np.ascontiguousarray(data))[None, None].to(device)
    autocast = inference_autocast(device)
    with torch.inference_mode(), autocast:
        output = model(tensor)
    values = output.to(device="cpu", dtype=torch.float32).reshape(-1).numpy()
    if values.shape != (MERLIN_EMBEDDING_DIMENSION,) or not np.isfinite(values).all():
        raise RuntimeValidationError(f"unexpected Merlin output shape: {values.shape}")
    return values
