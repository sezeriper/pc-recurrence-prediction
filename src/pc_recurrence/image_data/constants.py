from __future__ import annotations

from pathlib import Path

# Cleaned dataset wrapper used by the source workbook and DICOM folders.
DEFAULT_DATASET_ROOT = Path("dataset/Anonimleştirilmiş")
# Curated folder created by `pc-image-data preprocess`; all downstream tasks read from here.
DEFAULT_DICOM_ROOT = Path("outputs/dicom_selected")
# Untouched original anonymized folders; inventory and preprocess only read from here.
DEFAULT_SOURCE_DICOM_ROOT = DEFAULT_DATASET_ROOT / "dicom_anon"
DEFAULT_WORKBOOK = DEFAULT_DATASET_ROOT / "pankreas adeno ca 10 hasta.xlsx"
DEFAULT_SCAN_REVIEW_ROOT = Path("outputs/ct_series_review")
DEFAULT_SCAN_SELECTION = DEFAULT_SCAN_REVIEW_ROOT / "scan_selection.csv"
DEFAULT_INSPECTION_OUTPUT_ROOT = Path("outputs/dicom_inspection")
