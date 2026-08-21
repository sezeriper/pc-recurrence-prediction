from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import product

import nibabel as nib
import numpy as np
from nibabel.processing import resample_from_to, resample_to_output

from pc_recurrence.image_roi.constants import ROI_MARGIN_MM

from .constants import (
    MERLIN_HU_RANGE,
    MERLIN_INPUT_SIZE,
    MERLIN_TARGET_SPACING_MM,
    SPECTRE_CROP_SIZE,
    CenteringMode,
)


@dataclass(frozen=True)
class FoundationInput:
    data: np.ndarray
    grid_size: tuple[int, int, int]
    patch_starts: np.ndarray
    valid_voxel_count: int
    metadata: dict[str, object]


def _canonical_pair(
    ct_image: nib.Nifti1Image, mask_image: nib.Nifti1Image
) -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    ct = nib.as_closest_canonical(ct_image)
    mask = nib.as_closest_canonical(mask_image)
    if mask.shape != ct.shape or not np.allclose(mask.affine, ct.affine, atol=1e-4):
        mask = resample_from_to(mask, (ct.shape, ct.affine), order=0, mode="constant", cval=0)
    return ct, mask


def _validated_arrays(
    ct_image: nib.Nifti1Image, mask_image: nib.Nifti1Image
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ct, mask = _canonical_pair(ct_image, mask_image)
    data = np.asarray(ct.dataobj, dtype=np.float32)
    pancreas = np.asarray(mask.dataobj) > 0
    if data.ndim != 3 or any(size < 1 for size in data.shape):
        raise ValueError(f"ROI must be a non-empty 3D volume; received {data.shape}")
    if not np.isfinite(data).all():
        raise ValueError("ROI contains non-finite CT values")
    if not pancreas.any():
        raise ValueError("ROI pancreas mask is empty")
    return data, pancreas, np.asarray(ct.affine, dtype=np.float64)


def _mask_bounds(mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.argwhere(mask)
    start = coordinates.min(axis=0)
    stop = coordinates.max(axis=0) + 1
    center = (start + stop - 1) / 2.0
    return start, stop, center


def _volume_center(shape: tuple[int, ...] | np.ndarray) -> np.ndarray:
    """Geometric center of a volume in voxel coordinates."""
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
    pad_before = tuple(int(value) for value in destination_start)
    pad_after = tuple(int(value) for value in target - destination_stop)
    return np.ascontiguousarray(output), pad_before, pad_after


def prepare_spectre_input(
    ct_image: nib.Nifti1Image,
    mask_image: nib.Nifti1Image,
    *,
    margin_mm: float = ROI_MARGIN_MM,
    centering: CenteringMode = CenteringMode.VOLUME,
) -> FoundationInput:
    data, mask, affine = _validated_arrays(ct_image, mask_image)
    start, stop, mask_center = _mask_bounds(mask)
    spacing = nib.affines.voxel_sizes(affine)
    margin_voxels = np.ceil(margin_mm / spacing).astype(np.int64)
    required = stop - start + 2 * margin_voxels
    target_shape = tuple(
        max(crop, int(math.ceil(size / crop)) * crop)
        for size, crop in zip(required, SPECTRE_CROP_SIZE, strict=True)
    )
    center = (
        mask_center
        if centering is CenteringMode.PANCREAS
        else _volume_center(data.shape)
    )
    cropped, pad_before, pad_after = centered_crop_or_pad(
        data, center, target_shape, fill_value=-1000.0
    )
    grid = tuple(
        size // crop for size, crop in zip(target_shape, SPECTRE_CROP_SIZE, strict=True)
    )
    starts = np.asarray(
        [
            tuple(index * crop for index, crop in zip(indices, SPECTRE_CROP_SIZE, strict=True))
            for indices in product(*(range(size) for size in grid))
        ],
        dtype=np.int32,
    )
    return FoundationInput(
        data=cropped,
        grid_size=grid,
        patch_starts=starts,
        valid_voxel_count=int(np.prod(np.asarray(target_shape) - pad_before - pad_after)),
        metadata={
            "orientation": "RAS",
            "spacing_mm": [float(value) for value in spacing],
            "source_shape": list(data.shape),
            "centering": centering.value,
            "crop_center_voxel": center.tolist(),
            "pancreas_bbox_start": start.tolist(),
            "pancreas_bbox_stop": stop.tolist(),
            "pancreas_center_voxel": mask_center.tolist(),
            "volume_center_voxel": _volume_center(data.shape).tolist(),
            "target_shape": list(target_shape),
            "pad_before": list(pad_before),
            "pad_after": list(pad_after),
            "margin_mm": margin_mm,
            "fill_hu": -1000.0,
        },
    )


def prepare_merlin_input(
    ct_image: nib.Nifti1Image,
    mask_image: nib.Nifti1Image,
    *,
    centering: CenteringMode = CenteringMode.VOLUME,
) -> FoundationInput:
    canonical_ct, canonical_mask = _canonical_pair(ct_image, mask_image)
    resampled_ct = resample_to_output(
        canonical_ct,
        voxel_sizes=MERLIN_TARGET_SPACING_MM,
        order=1,
        mode="constant",
        cval=-1000.0,
    )
    resampled_mask = resample_from_to(
        canonical_mask,
        (resampled_ct.shape, resampled_ct.affine),
        order=0,
        mode="constant",
        cval=0,
    )
    data = np.asarray(resampled_ct.dataobj, dtype=np.float32)
    mask = np.asarray(resampled_mask.dataobj) > 0
    if not np.isfinite(data).all():
        raise ValueError("resampled ROI contains non-finite CT values")
    if not mask.any():
        raise ValueError("resampled ROI pancreas mask is empty")
    start, stop, mask_center = _mask_bounds(mask)
    center = (
        mask_center
        if centering is CenteringMode.PANCREAS
        else _volume_center(data.shape)
    )
    cropped, pad_before, pad_after = centered_crop_or_pad(
        data, center, MERLIN_INPUT_SIZE, fill_value=-1000.0
    )
    low, high = MERLIN_HU_RANGE
    cropped = np.clip(cropped, low, high)
    cropped = ((cropped - low) / (high - low)).astype(np.float32, copy=False)
    return FoundationInput(
        data=np.ascontiguousarray(cropped),
        grid_size=(1, 1, 1),
        patch_starts=np.zeros((1, 3), dtype=np.int32),
        valid_voxel_count=int(np.prod(np.asarray(MERLIN_INPUT_SIZE) - pad_before - pad_after)),
        metadata={
            "orientation": "RAS",
            "spacing_mm": list(MERLIN_TARGET_SPACING_MM),
            "resampled_shape": list(data.shape),
            "centering": centering.value,
            "crop_center_voxel": center.tolist(),
            "pancreas_bbox_start": start.tolist(),
            "pancreas_bbox_stop": stop.tolist(),
            "pancreas_center_voxel": mask_center.tolist(),
            "volume_center_voxel": _volume_center(data.shape).tolist(),
            "target_shape": list(MERLIN_INPUT_SIZE),
            "pad_before": list(pad_before),
            "pad_after": list(pad_after),
            "hu_clip": list(MERLIN_HU_RANGE),
            "intensity_range": [0.0, 1.0],
            "fill_hu": -1000.0,
        },
    )
