from __future__ import annotations

import nibabel as nib
import numpy as np

from pc_recurrence.image_embedding.constants import MERLIN_INPUT_SIZE, SPECTRE_CROP_SIZE
from pc_recurrence.image_embedding.foundation_preprocessing import (
    prepare_merlin_input,
    prepare_spectre_input,
)


def _off_center_ct() -> nib.Nifti1Image:
    affine = np.diag([2.0, 2.0, 2.5, 1.0])
    ct = np.full((28, 30, 20), -1000.0, dtype=np.float32)
    ct[2:6, 22:27, 3:7] = 1000.0
    return nib.Nifti1Image(ct, affine)


def _peak_center(data: np.ndarray) -> np.ndarray:
    return np.argwhere(data == data.max()).mean(axis=0)


def test_spectre_crop_uses_native_spacing_and_ct_volume_center() -> None:
    prepared = prepare_spectre_input(_off_center_ct())

    assert prepared.data.shape == SPECTRE_CROP_SIZE
    assert prepared.grid_size == (1, 1, 1)
    assert prepared.metadata["spacing_mm"] == [2.0, 2.0, 2.5]
    assert prepared.metadata["centering"] == "volume"
    np.testing.assert_allclose(
        prepared.metadata["crop_center_voxel"],
        prepared.metadata["volume_center_voxel"],
        atol=1e-9,
    )
    assert not np.allclose(
        _peak_center(prepared.data),
        (np.asarray(prepared.data.shape) - 1) / 2,
        atol=1.0,
    )


def test_merlin_crop_is_resampled_scaled_and_ct_volume_centered() -> None:
    prepared = prepare_merlin_input(_off_center_ct())

    assert prepared.data.shape == MERLIN_INPUT_SIZE
    assert prepared.data.dtype == np.float32
    assert float(prepared.data.min()) >= 0.0
    assert float(prepared.data.max()) <= 1.0
    assert prepared.metadata["spacing_mm"] == [1.5, 1.5, 3.0]
    assert prepared.metadata["centering"] == "volume"
    np.testing.assert_allclose(
        prepared.metadata["crop_center_voxel"],
        prepared.metadata["volume_center_voxel"],
        atol=1e-9,
    )
