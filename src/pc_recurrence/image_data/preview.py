from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pydicom


def render_dicom_series_preview(
    files: list[Path],
    output_path: Path,
    *,
    title: str,
) -> Path:
    """Render up to nine evenly spaced axial DICOM frames at WW/WL 400/40."""
    if not files:
        raise ValueError("cannot render a DICOM preview without files")
    indices = np.linspace(0, len(files) - 1, min(9, len(files)), dtype=int)
    figure, axes = plt.subplots(3, 3, figsize=(12, 13), constrained_layout=True)
    for plot in axes.flat:
        plot.set_axis_off()
    for plot, index in zip(axes.flat, indices, strict=False):
        path = files[int(index)]
        dataset = pydicom.dcmread(path, force=True)
        pixels = dataset.pixel_array.astype(np.float32)
        slope = float(getattr(dataset, "RescaleSlope", 1.0))
        intercept = float(getattr(dataset, "RescaleIntercept", 0.0))
        image_hu = pixels * slope + intercept
        plot.imshow(
            image_hu,
            cmap="gray",
            origin="lower",
            vmin=-160.0,
            vmax=240.0,
        )
        instance_number = str(getattr(dataset, "InstanceNumber", ""))
        plot.set_title(f"{path.name} | InstanceNumber {instance_number}", fontsize=8)
        plot.set_axis_off()
    figure.suptitle(f"{title}\nWW/WL 400/40", fontsize=10)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=150, facecolor="white")
    plt.close(figure)
    return output_path
