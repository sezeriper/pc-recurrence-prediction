from __future__ import annotations

from pathlib import Path

# The supplied ten-patient dataset contains the workbook and already-curated CT slices.
# This path is relative to the repository root, where the CLI commands are documented to run.
DEFAULT_DATASET_ROOT = Path("../Radyoloji Data/10Patients_Important_Slices")
# Curated folder created by `pc-image-data preprocess`; all downstream tasks read from here.
DEFAULT_DICOM_ROOT = Path("outputs/dicom_selected")
# The `IMG` folders are the source for direct copying. They are never modified.
DEFAULT_SOURCE_DICOM_ROOT = DEFAULT_DATASET_ROOT / "IMG"
DEFAULT_WORKBOOK = DEFAULT_DATASET_ROOT / "10Patients_Tabular.xlsx"
DEFAULT_INSPECTION_OUTPUT_ROOT = Path("outputs/dicom_inspection")
