from __future__ import annotations

from pathlib import Path

# All ROI runs center on the predicted pancreas; tumor-centered ROIs are not produced.
ROI_TARGET = "pancreas"

BUNDLE_REPOSITORY = "MONAI/pancreas_ct_dints_segmentation"
BUNDLE_VERSION = "0.5.2"
MODEL_REVISION = "1b4b04a0de2cf6236860891bf1c2e9494e1afaf7"
MODEL_FILENAME = "models/model.ts"
MODEL_SHA256 = "39D6087354FCFF7B90E27191E5654774B98E6D7B503AA5752EDFA9B07867BD5A"

EXPECTED_TORCH_VERSION = "2.10.0+rocm7.12.0a20260204"
EXPECTED_GPU_NAME = "AMD Radeon RX 6900 XT"
DEFAULT_RUNTIME_PYTHON = Path(r"D:\Projects\RCOm-windows-gfx1030\.venv-nightly\Scripts\python.exe")

ROI_SIZE = (96, 96, 96)
SW_BATCH_SIZE = 4
SW_OVERLAP = 0.625
TARGET_SPACING_MM = (1.0, 1.0, 1.0)
HU_RANGE = (-87.0, 199.0)
MIN_FREE_GPU_BYTES = 6 * 1024**3

MIN_TUMOR_VOLUME_MM3 = 100.0
MAX_PANCREAS_DISTANCE_MM = 5.0
ROI_MARGIN_MM = 15.0

# Cleaned dataset wrapper used by the source workbook and DICOM folders.
DEFAULT_DATASET_ROOT = Path("dataset/Anonimleştirilmiş")
# Curated folder created by `pc-image-roi preprocess`; all downstream tasks read from here.
DEFAULT_DICOM_ROOT = Path("outputs/dicom_selected")
# Untouched original anonymized folders; inventory and preprocess only read from here.
DEFAULT_SOURCE_DICOM_ROOT = DEFAULT_DATASET_ROOT / "dicom_anon"
DEFAULT_WORKBOOK = DEFAULT_DATASET_ROOT / "pankreas adeno ca 10 hasta.xlsx"
DEFAULT_SCAN_REVIEW_ROOT = Path("outputs/ct_series_review")
DEFAULT_SCAN_SELECTION = DEFAULT_SCAN_REVIEW_ROOT / "scan_selection.csv"
DEFAULT_OUTPUT_ROOT = Path("outputs/image_roi")
DEFAULT_MODEL_CACHE = Path(".cache/monai_models")
