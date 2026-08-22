from __future__ import annotations

import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pydicom
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

from .processing import BoundingBox, crop_affine, crop_array


def create_run_directory(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    base = datetime.now(UTC).strftime("run_%Y%m%dT%H%M%SZ")
    candidate = output_root / base
    suffix = 1
    while candidate.exists():
        candidate = output_root / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True)
    return candidate


def write_json(data: dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
    return path


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_summary(rows: list[dict[str, Any]], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "patient_id",
        "status",
        "study_uid",
        "reason",
        "series_uid",
        "dicom_file_count",
        "native_shape",
        "native_spacing_mm",
        "maximum_slice_gap_mm",
        "phase_status",
        "preprocessed_shape",
        "patch_count",
        "inference_seconds",
        "roi_volume_mm3",
        "patient_artifact_dir",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            serialized = row.copy()
            for key in ("native_shape", "native_spacing_mm", "preprocessed_shape"):
                value = serialized.get(key)
                if value is not None and not isinstance(value, str):
                    serialized[key] = "x".join(str(item) for item in value)
            writer.writerow(serialized)
    return path


def save_nifti(array: np.ndarray, affine: np.ndarray, path: Path, dtype: np.dtype) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.asarray(array, dtype=dtype)
    image = nib.Nifti1Image(data, affine)
    image.set_qform(affine, code=1)
    image.set_sform(affine, code=1)
    nib.save(image, path)
    return path


def save_case_artifacts(
    patient_dir: Path,
    volume_hu: np.ndarray,
    affine: np.ndarray,
    pancreas_mask: np.ndarray,
    bbox: BoundingBox,
    bbox_metadata: dict[str, Any],
) -> dict[str, str]:
    patient_dir.mkdir(parents=True, exist_ok=True)
    cropped_affine = crop_affine(affine, bbox.start)
    paths = {
        "pancreas_mask": save_nifti(
            pancreas_mask, affine, patient_dir / "pancreas_mask.nii.gz", np.uint8
        ),
        "roi_ct": save_nifti(
            crop_array(volume_hu, bbox), cropped_affine, patient_dir / "roi_ct.nii.gz", np.float32
        ),
        "roi_pancreas_mask": save_nifti(
            crop_array(pancreas_mask, bbox),
            cropped_affine,
            patient_dir / "roi_pancreas_mask.nii.gz",
            np.uint8,
        ),
    }
    bbox_path = write_json(bbox_metadata, patient_dir / "bbox.json")
    montage_path = render_review_montage(
        volume_hu,
        affine,
        pancreas_mask,
        bbox,
        patient_dir / "review_montage.png",
    )
    return {
        **{key: str(path.name) for key, path in paths.items()},
        "bbox": bbox_path.name,
        "review_montage": montage_path.name,
    }


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


def _canonical_data(array: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = nib.as_closest_canonical(nib.Nifti1Image(array, affine))
    return np.asarray(image.dataobj), image.affine


def _plane(array: np.ndarray, axis: int, index: int) -> np.ndarray:
    return np.take(array, index, axis=axis).T


def render_review_montage(
    volume_hu: np.ndarray,
    affine: np.ndarray,
    pancreas_mask: np.ndarray,
    bbox: BoundingBox,
    output_path: Path,
) -> Path:
    canonical_ct, canonical_affine = _canonical_data(volume_hu.astype(np.float32), affine)
    canonical_pancreas, _ = _canonical_data(pancreas_mask.astype(np.uint8), affine)
    bbox_mask = np.zeros_like(pancreas_mask, dtype=np.uint8)
    bbox_mask[
        bbox.start[0] : bbox.stop[0],
        bbox.start[1] : bbox.stop[1],
        bbox.start[2] : bbox.stop[2],
    ] = 1
    canonical_bbox, _ = _canonical_data(bbox_mask, affine)
    bbox_coordinates = np.argwhere(canonical_bbox > 0)
    bbox_start = bbox_coordinates.min(axis=0)
    bbox_stop = bbox_coordinates.max(axis=0) + 1
    spacing = nib.affines.voxel_sizes(canonical_affine)

    specifications = [
        (2, "Axial", ("L", "R", "P", "A"), spacing[1] / spacing[0]),
        (1, "Coronal", ("L", "R", "I", "S"), spacing[2] / spacing[0]),
        (0, "Sagittal", ("P", "A", "I", "S"), spacing[2] / spacing[1]),
    ]
    figure, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    pancreas_cmap = ListedColormap([(0.0, 1.0, 1.0, 0.30)])
    for plot, (axis, title, labels, aspect) in zip(axes, specifications, strict=True):
        reduce_axes = tuple(index for index in range(3) if index != axis)
        slice_scores = np.sum(canonical_pancreas > 0, axis=reduce_axes)
        index = int(np.argmax(slice_scores))
        ct_slice = np.clip(_plane(canonical_ct, axis, index), -160.0, 240.0)
        plot.imshow(ct_slice, cmap="gray", origin="lower", vmin=-160.0, vmax=240.0, aspect=aspect)
        pancreas_slice = _plane(canonical_pancreas, axis, index) > 0
        plot.imshow(
            np.ma.masked_where(~pancreas_slice, pancreas_slice),
            cmap=pancreas_cmap,
            origin="lower",
            aspect=aspect,
        )

        displayed_axes = [dimension for dimension in range(3) if dimension != axis]
        horizontal, vertical = displayed_axes[0], displayed_axes[1]
        rectangle = Rectangle(
            (bbox_start[horizontal] - 0.5, bbox_start[vertical] - 0.5),
            bbox_stop[horizontal] - bbox_start[horizontal],
            bbox_stop[vertical] - bbox_start[vertical],
            linewidth=2,
            edgecolor="yellow",
            facecolor="none",
        )
        plot.add_patch(rectangle)
        left, right, bottom, top = labels
        plot.text(0.01, 0.5, left, color="white", weight="bold", transform=plot.transAxes)
        plot.text(
            0.99, 0.5, right, color="white", weight="bold", ha="right", transform=plot.transAxes
        )
        plot.text(
            0.5, 0.01, bottom, color="white", weight="bold", ha="center", transform=plot.transAxes
        )
        plot.text(
            0.5,
            0.99,
            top,
            color="white",
            weight="bold",
            ha="center",
            va="top",
            transform=plot.transAxes,
        )
        plot.set_title(f"{title} (slice {index})")
        plot.set_axis_off()
    figure.suptitle("Pancreas (cyan), 15 mm ROI (yellow) | WW/WL 400/40")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, facecolor="black")
    plt.close(figure)
    return output_path
