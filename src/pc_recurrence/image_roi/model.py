from __future__ import annotations

import hashlib
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from monai.inferers import sliding_window_inference
from nibabel.processing import resample_from_to, resample_to_output

from .constants import (
    BUNDLE_REPOSITORY,
    EXPECTED_GPU_NAME,
    EXPECTED_TORCH_VERSION,
    HU_RANGE,
    MIN_FREE_GPU_BYTES,
    MODEL_FILENAME,
    MODEL_REVISION,
    MODEL_SHA256,
    ROI_SIZE,
    SW_BATCH_SIZE,
    SW_OVERLAP,
    TARGET_SPACING_MM,
)


class RuntimeValidationError(RuntimeError):
    """Raised when the required ROCm runtime is unavailable or has changed."""


@dataclass(frozen=True)
class RuntimeInfo:
    torch_version: str
    hip_version: str | None
    device_index: int
    device_name: str
    total_gpu_bytes: int
    free_gpu_bytes: int
    smoke_test_seconds: float
    smoke_test_shape: tuple[int, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.__dict__,
            "smoke_test_shape": list(self.smoke_test_shape),
        }


@dataclass
class SegmentationResult:
    labels_original: np.ndarray
    preprocessed_shape: tuple[int, int, int]
    patch_count: int
    inference_seconds: float


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def acquire_model(cache_dir: Path, *, local_files_only: bool = False) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = Path(
        hf_hub_download(
            repo_id=BUNDLE_REPOSITORY,
            filename=MODEL_FILENAME,
            revision=MODEL_REVISION,
            local_dir=cache_dir / "pancreas_ct_dints_segmentation",
            local_files_only=local_files_only,
        )
    )
    actual = sha256_file(path)
    if actual != MODEL_SHA256:
        raise RuntimeValidationError(
            f"model checksum mismatch: expected {MODEL_SHA256}, received {actual}"
        )
    return path


def _memory_info(device_index: int) -> tuple[int, int]:
    with torch.cuda.device(device_index):
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes), int(total_bytes)


def validate_and_load_runtime(model_path: Path) -> tuple[torch.jit.ScriptModule, RuntimeInfo]:
    if torch.__version__ != EXPECTED_TORCH_VERSION:
        raise RuntimeValidationError(
            f"expected torch {EXPECTED_TORCH_VERSION}, found {torch.__version__}"
        )
    if not torch.cuda.is_available():
        raise RuntimeValidationError("ROCm GPU is unavailable; CPU fallback is prohibited")
    if torch.cuda.device_count() < 1:
        raise RuntimeValidationError("no ROCm devices found")
    device_index = 0
    device_name = torch.cuda.get_device_name(device_index)
    if device_name != EXPECTED_GPU_NAME:
        raise RuntimeValidationError(f"expected {EXPECTED_GPU_NAME}, found {device_name}")
    free_bytes, total_bytes = _memory_info(device_index)
    if free_bytes < MIN_FREE_GPU_BYTES:
        raise RuntimeValidationError(
            f"at least 6 GiB free GPU memory is required; found {free_bytes / 1024**3:.2f} GiB"
        )
    actual = sha256_file(model_path)
    if actual != MODEL_SHA256:
        raise RuntimeValidationError(
            f"model checksum mismatch: expected {MODEL_SHA256}, received {actual}"
        )

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    model = torch.jit.load(str(model_path), map_location=device).eval()
    smoke = torch.zeros((SW_BATCH_SIZE, 1, *ROI_SIZE), dtype=torch.float32, device=device)
    started = time.perf_counter()
    with torch.inference_mode():
        output = model(smoke)
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    expected_shape = (SW_BATCH_SIZE, 3, *ROI_SIZE)
    if tuple(output.shape) != expected_shape or not bool(torch.isfinite(output).all().item()):
        raise RuntimeValidationError(
            f"model smoke test failed: shape={tuple(output.shape)}, finite="
            f"{bool(torch.isfinite(output).all().item())}"
        )
    del smoke, output
    free_after, total_after = _memory_info(device_index)
    return model, RuntimeInfo(
        torch_version=torch.__version__,
        hip_version=getattr(torch.version, "hip", None),
        device_index=device_index,
        device_name=device_name,
        total_gpu_bytes=total_after,
        free_gpu_bytes=free_after,
        smoke_test_seconds=elapsed,
        smoke_test_shape=expected_shape,
    )


def preprocess_ct(volume_hu: np.ndarray, affine_ras: np.ndarray) -> nib.Nifti1Image:
    source = nib.Nifti1Image(volume_hu.astype(np.float32, copy=False), affine_ras)
    resampled = resample_to_output(source, voxel_sizes=TARGET_SPACING_MM, order=1, mode="nearest")
    data = np.asarray(resampled.dataobj, dtype=np.float32)
    low, high = HU_RANGE
    data = np.clip(data, low, high)
    data = (data - low) / (high - low)
    return nib.Nifti1Image(data.astype(np.float32, copy=False), resampled.affine)


def _patch_count(
    shape: tuple[int, int, int],
    roi_size: tuple[int, int, int] = ROI_SIZE,
    overlap: float = SW_OVERLAP,
) -> int:
    counts = []
    for size, roi in zip(shape, roi_size, strict=True):
        if size <= roi:
            counts.append(1)
            continue
        interval = max(int(roi * (1.0 - overlap)), 1)
        counts.append(math.ceil((size - roi) / interval) + 1)
    return math.prod(counts)


def segment_volume(
    model: torch.jit.ScriptModule,
    volume_hu: np.ndarray,
    affine_ras: np.ndarray,
    *,
    inference_device: torch.device | str = "cuda:0",
    output_device: torch.device | str = "cpu",
    roi_size: tuple[int, int, int] = ROI_SIZE,
    sw_batch_size: int = SW_BATCH_SIZE,
    overlap: float = SW_OVERLAP,
) -> SegmentationResult:
    preprocessed = preprocess_ct(volume_hu, affine_ras)
    data = np.asarray(preprocessed.dataobj, dtype=np.float32)
    tensor = torch.from_numpy(np.ascontiguousarray(data))[None, None]
    started = time.perf_counter()
    with torch.inference_mode():
        logits = sliding_window_inference(
            inputs=tensor,
            roi_size=roi_size,
            sw_batch_size=sw_batch_size,
            predictor=model,
            overlap=overlap,
            mode="constant",
            sw_device=torch.device(inference_device),
            device=torch.device(output_device),
            progress=False,
        )
    elapsed = time.perf_counter() - started
    if not bool(torch.isfinite(logits).all().item()):
        raise RuntimeValidationError("non-finite values in segmentation logits")
    labels = torch.argmax(logits, dim=1)[0].to(torch.uint8).numpy()
    label_image = nib.Nifti1Image(labels, preprocessed.affine)
    restored = resample_from_to(
        label_image,
        (volume_hu.shape, affine_ras),
        order=0,
        mode="constant",
        cval=0,
    )
    labels_original = np.rint(np.asarray(restored.dataobj)).astype(np.uint8)
    return SegmentationResult(
        labels_original=labels_original,
        preprocessed_shape=tuple(int(value) for value in data.shape),
        patch_count=_patch_count(
            tuple(int(value) for value in data.shape), roi_size=roi_size, overlap=overlap
        ),
        inference_seconds=elapsed,
    )
