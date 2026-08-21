from __future__ import annotations

import nibabel as nib
import numpy as np

from pc_recurrence.image_embedding.constants import MERLIN_INPUT_SIZE, SPECTRE_CROP_SIZE
from pc_recurrence.image_embedding.foundation_preprocessing import (
    prepare_merlin_input,
    prepare_spectre_input,
)


def _off_center_case() -> tuple[nib.Nifti1Image, nib.Nifti1Image]:
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    ct = np.full((28, 30, 20), -1000.0, dtype=np.float32)
    mask = np.zeros_like(ct, dtype=np.uint8)
    mask[2:6, 22:27, 3:7] = 1
    ct[mask > 0] = 1000.0
    return nib.Nifti1Image(ct, affine), nib.Nifti1Image(mask, affine)


def _peak_center(data: np.ndarray) -> np.ndarray:
    return np.argwhere(data == data.max()).mean(axis=0)


def test_spectre_crop_is_native_spacing_centered_and_grid_padded() -> None:
    ct, mask = _off_center_case()

    prepared = prepare_spectre_input(ct, mask)

    assert all(
        size % crop == 0
        for size, crop in zip(prepared.data.shape, SPECTRE_CROP_SIZE, strict=True)
    )
    assert prepared.metadata["spacing_mm"] == [2.0, 2.0, 2.5]
    np.testing.assert_allclose(
        _peak_center(prepared.data),
        (np.asarray(prepared.data.shape) - 1) / 2,
        atol=1.0,
    )


def test_merlin_crop_is_resampled_scaled_and_centered() -> None:
    ct, mask = _off_center_case()

    prepared = prepare_merlin_input(ct, mask)

    assert prepared.data.shape == MERLIN_INPUT_SIZE
    assert prepared.data.dtype == np.float32
    assert float(prepared.data.min()) >= 0.0
    assert float(prepared.data.max()) <= 1.0
    assert prepared.metadata["spacing_mm"] == [1.5, 1.5, 3.0]
    np.testing.assert_allclose(
        _peak_center(prepared.data),
        (np.asarray(MERLIN_INPUT_SIZE) - 1) / 2,
        atol=2.0,
    )
