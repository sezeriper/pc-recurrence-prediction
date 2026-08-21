from __future__ import annotations

import numpy as np
import pytest

from pc_recurrence.image_roi.processing import (
    crop_affine,
    expanded_bounding_box,
    select_tumor_component,
)


def _affine(spacing: tuple[float, float, float]) -> np.ndarray:
    return np.diag([*spacing, 1.0]).astype(np.float64)


def test_component_filters_and_keeps_largest_qualifying() -> None:
    labels = np.zeros((30, 30, 30), dtype=np.uint8)
    labels[4:15, 4:15, 4:15] = 1
    labels[15:21, 6:12, 6:12] = 2  # 216 mm3 and adjacent to pancreas
    labels[7:12, 15:20, 6:12] = 2  # 150 mm3 and adjacent, but smaller
    labels[25:29, 25:29, 25:29] = 2  # 64 mm3 and too small/far

    result = select_tumor_component(labels, _affine((1, 1, 1)))

    assert result.selected_volume_mm3 == pytest.approx(216.0)
    assert np.count_nonzero(result.tumor_mask) == 216
    reasons = {decision.reason for decision in result.decisions}
    assert "qualifying" in reasons
    assert "below_minimum_volume" in reasons


def test_empty_pancreas_rejects_tumor() -> None:
    labels = np.zeros((20, 20, 20), dtype=np.uint8)
    labels[2:12, 2:12, 2:12] = 2
    result = select_tumor_component(labels, _affine((1, 1, 1)))
    assert result.selected_component is None
    assert result.decisions[0].reason == "pancreas_not_detected"


def test_distance_threshold_uses_physical_spacing() -> None:
    labels = np.zeros((20, 20, 20), dtype=np.uint8)
    labels[1:5, 1:5, 1:5] = 1
    labels[7:12, 1:6, 1:6] = 2
    result = select_tumor_component(labels, _affine((2, 1, 1)))
    assert result.selected_component is None
    assert result.decisions[0].minimum_pancreas_distance_mm == pytest.approx(6.0)
    assert result.decisions[0].reason == "too_far_from_pancreas"


def test_bbox_expands_by_ceiling_of_15mm_and_clamps() -> None:
    mask = np.zeros((20, 40, 30), dtype=bool)
    mask[2:5, 20:23, 10:12] = True
    affine = _affine((2.0, 0.8, 2.5))

    bbox = expanded_bounding_box(mask, affine)

    assert bbox.start == (0, 1, 4)
    assert bbox.stop == (13, 40, 18)
    assert bbox.achieved_margin_low_mm == pytest.approx((4.0, 15.2, 15.0))
    assert bbox.achieved_margin_high_mm == pytest.approx((16.0, 13.6, 15.0))


def test_crop_affine_preserves_world_location() -> None:
    affine = _affine((2.0, 3.0, 4.0))
    affine[:3, 3] = [10, 20, 30]
    start = (2, 3, 4)
    cropped = crop_affine(affine, start)
    assert cropped[:3, 3].tolist() == pytest.approx([14, 29, 46])
