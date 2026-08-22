from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image
from pydicom.uid import generate_uid
from test_image_dicom import _write_slice

from pc_recurrence.image_roi.artifacts import (
    render_dicom_series_preview,
    save_case_artifacts,
)
from pc_recurrence.image_roi.processing import expanded_bounding_box


def test_detected_case_artifacts_are_aligned_and_rendered(tmp_path: Path) -> None:
    shape = (30, 40, 20)
    ct = np.zeros(shape, dtype=np.float32)
    pancreas = np.zeros(shape, dtype=bool)
    tumor = np.zeros(shape, dtype=bool)
    pancreas[5:25, 8:32, 4:16] = True
    tumor[12:17, 18:24, 8:12] = True
    affine = np.diag([1.0, 1.0, 2.0, 1.0])
    bbox = expanded_bounding_box(pancreas, affine)

    artifacts = save_case_artifacts(
        tmp_path,
        ct,
        affine,
        pancreas,
        bbox,
        bbox.to_dict(),
    )

    assert set(artifacts) == {
        "pancreas_mask",
        "roi_ct",
        "roi_pancreas_mask",
        "bbox",
        "review_montage",
    }
    full_pancreas = nib.load(tmp_path / "pancreas_mask.nii.gz")
    roi_pancreas = nib.load(tmp_path / "roi_pancreas_mask.nii.gz")
    assert full_pancreas.shape == shape
    expected_roi_shape = tuple(high - low for low, high in zip(bbox.start, bbox.stop, strict=True))
    assert roi_pancreas.shape == expected_roi_shape
    expected_origin = (affine @ np.asarray([*bbox.start, 1.0]))[:3]
    assert np.allclose(roi_pancreas.affine[:3, 3], expected_origin)
    assert not (tmp_path / "tumor_mask.nii.gz").exists()
    assert not (tmp_path / "roi_tumor_mask.nii.gz").exists()
    with Image.open(tmp_path / "review_montage.png") as montage:
        assert montage.width > montage.height
        assert montage.width >= 1000


def test_dicom_series_preview_is_labeled_three_by_three_montage(tmp_path: Path) -> None:
    series_uid = generate_uid()
    files: list[Path] = []
    for index in range(1, 11):
        path = tmp_path / f"IM{index:03d}.dcm"
        _write_slice(
            path,
            series_uid=series_uid,
            z=float(index),
            stored_value=index,
            instance_number=index,
        )
        files.append(path)
    output = render_dicom_series_preview(
        files, tmp_path / "preview.png", title="Patient 1 / candidate"
    )
    with Image.open(output) as preview:
        assert preview.width > 1000
        assert preview.height > 1000
        assert 0.75 < preview.width / preview.height < 1.1
