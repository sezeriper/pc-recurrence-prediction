from __future__ import annotations

from pathlib import Path

from PIL import Image
from pydicom.uid import generate_uid
from test_image_dicom import _write_slice

from pc_recurrence.image_data.preview import render_dicom_series_preview


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
