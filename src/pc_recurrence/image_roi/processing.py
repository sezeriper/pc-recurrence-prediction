from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from .constants import MAX_PANCREAS_DISTANCE_MM, MIN_TUMOR_VOLUME_MM3, ROI_MARGIN_MM


@dataclass(frozen=True)
class ComponentDecision:
    component: int
    voxel_count: int
    volume_mm3: float
    minimum_pancreas_distance_mm: float | None
    accepted: bool
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass(frozen=True)
class TumorSelection:
    pancreas_mask: np.ndarray
    tumor_mask: np.ndarray
    decisions: list[ComponentDecision]
    selected_component: int | None
    selected_volume_mm3: float | None


@dataclass(frozen=True)
class BoundingBox:
    start: tuple[int, int, int]
    stop: tuple[int, int, int]
    tumor_start: tuple[int, int, int]
    tumor_stop: tuple[int, int, int]
    spacing_mm: tuple[float, float, float]
    requested_margin_mm: float
    achieved_margin_low_mm: tuple[float, float, float]
    achieved_margin_high_mm: tuple[float, float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "index_convention": "start inclusive, stop exclusive",
            "start": list(self.start),
            "stop": list(self.stop),
            "tumor_start": list(self.tumor_start),
            "tumor_stop": list(self.tumor_stop),
            "spacing_mm": list(self.spacing_mm),
            "requested_margin_mm": self.requested_margin_mm,
            "achieved_margin_low_mm": list(self.achieved_margin_low_mm),
            "achieved_margin_high_mm": list(self.achieved_margin_high_mm),
        }


def voxel_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple(float(np.linalg.norm(affine[:3, axis])) for axis in range(3))


def voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))


def select_tumor_component(
    labels: np.ndarray,
    affine: np.ndarray,
    *,
    minimum_volume_mm3: float = MIN_TUMOR_VOLUME_MM3,
    maximum_pancreas_distance_mm: float = MAX_PANCREAS_DISTANCE_MM,
) -> TumorSelection:
    pancreas = labels == 1
    tumor_candidates = labels == 2
    components, count = ndimage.label(
        tumor_candidates, structure=np.ones((3, 3, 3), dtype=np.uint8)
    )
    spacing = voxel_spacing(affine)
    volume_per_voxel = voxel_volume_mm3(affine)
    distance = (
        ndimage.distance_transform_edt(~pancreas, sampling=spacing) if np.any(pancreas) else None
    )
    decisions: list[ComponentDecision] = []
    accepted: list[tuple[float, int]] = []
    for component in range(1, count + 1):
        mask = components == component
        voxels = int(np.count_nonzero(mask))
        volume = voxels * volume_per_voxel
        minimum_distance = float(np.min(distance[mask])) if distance is not None else None
        if volume < minimum_volume_mm3:
            reason = "below_minimum_volume"
            is_accepted = False
        elif distance is None:
            reason = "pancreas_not_detected"
            is_accepted = False
        elif minimum_distance is None or minimum_distance > maximum_pancreas_distance_mm:
            reason = "too_far_from_pancreas"
            is_accepted = False
        else:
            reason = "qualifying"
            is_accepted = True
            accepted.append((volume, component))
        decisions.append(
            ComponentDecision(
                component=component,
                voxel_count=voxels,
                volume_mm3=volume,
                minimum_pancreas_distance_mm=minimum_distance,
                accepted=is_accepted,
                reason=reason,
            )
        )
    selected_component = max(accepted)[1] if accepted else None
    selected = (
        components == selected_component
        if selected_component is not None
        else np.zeros_like(labels, bool)
    )
    selected_volume = max(accepted)[0] if accepted else None
    return TumorSelection(
        pancreas_mask=pancreas,
        tumor_mask=selected,
        decisions=decisions,
        selected_component=selected_component,
        selected_volume_mm3=selected_volume,
    )


def expanded_bounding_box(
    tumor_mask: np.ndarray,
    affine: np.ndarray,
    *,
    margin_mm: float = ROI_MARGIN_MM,
) -> BoundingBox:
    coordinates = np.argwhere(tumor_mask)
    if coordinates.size == 0:
        raise ValueError("cannot create a bounding box for an empty tumor mask")
    tumor_start_array = coordinates.min(axis=0)
    tumor_stop_array = coordinates.max(axis=0) + 1
    spacing = voxel_spacing(affine)
    margin_voxels = np.asarray(
        [math.ceil(margin_mm / axis_spacing) for axis_spacing in spacing], dtype=np.int64
    )
    start = np.maximum(tumor_start_array - margin_voxels, 0)
    stop = np.minimum(tumor_stop_array + margin_voxels, np.asarray(tumor_mask.shape))
    low = (tumor_start_array - start) * np.asarray(spacing)
    high = (stop - tumor_stop_array) * np.asarray(spacing)
    return BoundingBox(
        start=tuple(int(value) for value in start),
        stop=tuple(int(value) for value in stop),
        tumor_start=tuple(int(value) for value in tumor_start_array),
        tumor_stop=tuple(int(value) for value in tumor_stop_array),
        spacing_mm=spacing,
        requested_margin_mm=margin_mm,
        achieved_margin_low_mm=tuple(float(value) for value in low),
        achieved_margin_high_mm=tuple(float(value) for value in high),
    )


def crop_affine(affine: np.ndarray, start: tuple[int, int, int]) -> np.ndarray:
    cropped = affine.copy()
    cropped[:3, 3] = (affine @ np.asarray([*start, 1.0], dtype=np.float64))[:3]
    return cropped


def crop_array(array: np.ndarray, bbox: BoundingBox) -> np.ndarray:
    slices = tuple(slice(low, high) for low, high in zip(bbox.start, bbox.stop, strict=True))
    return np.asarray(array[slices])
