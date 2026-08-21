from __future__ import annotations

import numpy as np
import torch

from pc_recurrence.image_roi.model import _patch_count, preprocess_ct, segment_volume


class ThresholdPredictor(torch.nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        background = 0.5 - value
        pancreas = torch.full_like(value, -1.0)
        tumor = value - 0.5
        return torch.cat([background, pancreas, tumor], dim=1)


def test_preprocessing_clips_and_scales_without_truncation() -> None:
    volume = np.asarray([[[-100.0, -87.0, 56.0, 199.0, 300.0]]], dtype=np.float32)
    image = preprocess_ct(volume, np.eye(4))
    data = np.asarray(image.dataobj)
    assert data.shape == volume.shape
    assert data[0, 0, 0] == 0.0
    assert data[0, 0, 1] == 0.0
    assert data[0, 0, 3] == 1.0
    assert data[0, 0, 4] == 1.0


def test_patch_count_matches_expected_overlap_grid() -> None:
    assert _patch_count((96, 96, 96)) == 1
    assert _patch_count((97, 132, 168)) == 2 * 2 * 3


def test_cpu_reference_inference_handles_multiple_overlapping_chunks() -> None:
    volume = np.full((12, 10, 8), 199.0, dtype=np.float32)
    result = segment_volume(
        ThresholdPredictor().eval(),
        volume,
        np.eye(4),
        inference_device="cpu",
        output_device="cpu",
        roi_size=(6, 6, 6),
        sw_batch_size=2,
        overlap=0.5,
    )
    assert result.labels_original.shape == volume.shape
    assert np.all(result.labels_original == 2)
    assert result.patch_count == 3 * 3 * 2
