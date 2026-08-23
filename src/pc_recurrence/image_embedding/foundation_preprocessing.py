from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import nibabel as nib
import numpy as np
from nibabel.affines import voxel_sizes
from nibabel.processing import resample_to_output

from .constants import (
    MERLIN_HU_RANGE,
    MERLIN_INPUT_SIZE,
    MERLIN_TARGET_SPACING_MM,
    SPECTRE_CROP_SIZE,
)


@dataclass(frozen=True)
class FoundationInput:
    data: np.ndarray
    grid_size: tuple[int, int, int]
    patch_starts: np.ndarray
    valid_voxel_count: int
    metadata: dict[str, object]


def _canonical_ct(ct_image: nib.Nifti1Image) -> nib.Nifti1Image:
    ct = nib.as_closest_canonical(ct_image)
    data = np.asarray(ct.dataobj, dtype=np.float32)
    if data.ndim != 3 or any(size < 1 for size in data.shape):
        raise ValueError(f"CT must be a non-empty 3D volume; received {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("CT contains non-finite values")
    return ct


def _volume_center(shape: tuple[int, ...] | np.ndarray) -> np.ndarray:
    """Geometric center of a selected CT volume in voxel coordinates."""
    return (np.asarray(shape, dtype=np.float64) - 1) / 2.0


def centered_crop_or_pad(
    data: np.ndarray,
    center: np.ndarray,
    target_shape: tuple[int, int, int],
    *,
    fill_value: float,
) -> tuple[np.ndarray, tuple[int, int, int], tuple[int, int, int]]:
    source_shape = np.asarray(data.shape, dtype=np.int64)
    target = np.asarray(target_shape, dtype=np.int64)
    requested_start = np.floor(center - (target - 1) / 2.0).astype(np.int64)
    requested_stop = requested_start + target
    source_start = np.maximum(requested_start, 0)
    source_stop = np.minimum(requested_stop, source_shape)
    destination_start = source_start - requested_start
    destination_stop = destination_start + (source_stop - source_start)
    output = np.full(target_shape, fill_value, dtype=np.float32)
    source_slices = tuple(
        slice(int(low), int(high))
        for low, high in zip(source_start, source_stop, strict=True)
    )
    destination_slices = tuple(
        slice(int(low), int(high))
        for low, high in zip(destination_start, destination_stop, strict=True)
    )
    output[destination_slices] = data[source_slices]
    pad_before = cast(tuple[int, int, int], tuple(int(value) for value in destination_start))
    pad_after = cast(tuple[int, int, int], tuple(int(value) for value in target - destination_stop))
    return np.ascontiguousarray(output), pad_before, pad_after


def prepare_spectre_input(ct_image: nib.Nifti1Image) -> FoundationInput:
    ct = _canonical_ct(ct_image)
    data = np.asarray(ct.dataobj, dtype=np.float32)
    affine = np.asarray(ct.affine, dtype=np.float64)
    center = _volume_center(data.shape)
    cropped, pad_before, pad_after = centered_crop_or_pad(
        data, center, SPECTRE_CROP_SIZE, fill_value=-1000.0
    )
    metadata: dict[str, object] = {
        "orientation": "RAS",
        "spacing_mm": [float(value) for value in voxel_sizes(affine)],
        "source_shape": list(data.shape),
        "centering": "volume",
        "volume_center_voxel": center.tolist(),
        "crop_center_voxel": center.tolist(),
        "target_shape": list(SPECTRE_CROP_SIZE),
        "pad_before": list(pad_before),
        "pad_after": list(pad_after),
        "fill_hu": -1000.0,
    }
    valid_shape = np.asarray(SPECTRE_CROP_SIZE) - pad_before - pad_after
    return FoundationInput(
        data=cropped,
        grid_size=(1, 1, 1),
        patch_starts=np.zeros((1, 3), dtype=np.int32),
        valid_voxel_count=int(np.prod(valid_shape)),
        metadata=metadata,
    )


def prepare_merlin_input(ct_image: nib.Nifti1Image) -> FoundationInput:
    canonical_ct = _canonical_ct(ct_image)
    resampled_ct = resample_to_output(
        canonical_ct,
        voxel_sizes=MERLIN_TARGET_SPACING_MM,
        order=1,
        mode="constant",
        cval=-1000.0,
    )
    data = np.asarray(resampled_ct.dataobj, dtype=np.float32)
    if not np.isfinite(data).all():
        raise ValueError("resampled CT contains non-finite values")
    center = _volume_center(data.shape)
    cropped, pad_before, pad_after = centered_crop_or_pad(
        data, center, MERLIN_INPUT_SIZE, fill_value=-1000.0
    )
    low, high = MERLIN_HU_RANGE
    cropped = np.clip(cropped, low, high)
    cropped = ((cropped - low) / (high - low)).astype(np.float32, copy=False)
    metadata: dict[str, object] = {
        "orientation": "RAS",
        "spacing_mm": list(MERLIN_TARGET_SPACING_MM),
        "resampled_shape": list(data.shape),
        "centering": "volume",
        "volume_center_voxel": center.tolist(),
        "crop_center_voxel": center.tolist(),
        "target_shape": list(MERLIN_INPUT_SIZE),
        "pad_before": list(pad_before),
        "pad_after": list(pad_after),
        "hu_clip": list(MERLIN_HU_RANGE),
        "intensity_range": [0.0, 1.0],
        "fill_hu": -1000.0,
    }
    valid_shape = np.asarray(MERLIN_INPUT_SIZE) - pad_before - pad_after
    return FoundationInput(
        data=np.ascontiguousarray(cropped),
        grid_size=(1, 1, 1),
        patch_starts=np.zeros((1, 3), dtype=np.int32),
        valid_voxel_count=int(np.prod(valid_shape)),
        metadata=metadata,
    )
