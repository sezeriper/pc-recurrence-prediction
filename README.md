# Pancreatic Cancer Image Preprocessing

This repository implements the image modality of the pancreatic adenocarcinoma recurrence
project: workbook-driven DICOM curation, MONAI pancreas ROI detection, and frozen SPECTRE-Large /
Merlin image embeddings. It reads each approved Turkish spreadsheet row's `Görüntü alanı` slice
range, copies the selected slices into a curated folder, segments the pancreas, and encodes the
pancreas-centered ROIs.

The included labels and splits are deliberately provisional pipeline-smoke inputs, not a
scientific recurrence model or evaluation.

## Setup

Install local CPU imaging dependencies for development and tests with:

```powershell
uv sync --extra imaging --group dev
```

Real inference is dispatched to the verified RX 6900 XT ROCm interpreter at
`D:\Projects\RCOm-windows-gfx1030\.venv-nightly`. Install this project into that environment while
protecting its accelerator packages:

```powershell
uv pip install `
  --python "D:\Projects\RCOm-windows-gfx1030\.venv-nightly\Scripts\python.exe" `
  --constraint requirements-rocm-constraints.txt `
  --editable ".[imaging]"
```

## CT segmentation and ROI review

The image pipeline uses the pinned MONAI `pancreas_ct_dints_segmentation` TorchScript bundle.

Create the curated slice folder first; every image task reads from it by default afterwards:

```powershell
uv run pc-image-roi preprocess
```

The preprocess command reads `Görüntü alanı` from the workbook as one-based inclusive ordinals
after ascending DICOM `InstanceNumber`, copies the selected slices from the untouched originals
under `dataset/dicom_anon/PATIENT<hasta no>` into `dataset/dicom_selected/PATIENT<hasta no>`, and
records every copy in `dataset/dicom_selected/curation_manifest.json`. Re-running is idempotent
(byte-identical files are skipped; `--force` re-copies). Patients whose folder or range is missing
or whose range is invalid are recorded as skips without aborting the run; geometry eligibility is
documented per patient but does not gate the copy. `--patients "Patient 1,PATIENT853534"` accepts
workbook IDs or DICOM folder names for a subset.

Audit DICOM geometry without inference, run using an already cached model, or acquire the pinned
model and run the complete pipeline. These commands default to the curated folder, which already
contains exactly the table-defined slices, and reorder them by physical position for loading. Every
slice range is accepted regardless of slice-position gaps or other geometry anomalies; anomalies
are recorded as per-patient warnings instead of skip reasons. The workbook's `hasta no` column
links each row to its anonymized CT folder; the image pipeline resolves that mapping automatically
and keeps the workbook `hasta id` (`Patient N`) as the aligned patient ID in all downstream
artifacts.

```powershell
uv run pc-image-roi inspect
uv run pc-image-roi segment `
  --run-dir "outputs/image_roi/<existing-run>" --resume
uv run pc-image-roi run
```

Use `--patients "Patient 1,Patient 3"` (workbook IDs) or `--patients "PATIENT2321275"` (DICOM
folder names) for a subset. `--force` replaces cached state within the selected run. There is no
CPU fallback: segmentation stops if the exact ROCm build, RX 6900 XT, or minimum free-memory
requirement is unavailable. In the current pilot, Patient 1 has no CT folder and Patient 10 is
accepted despite the 64 mm gap between its two acquisition blocks, so 9 of 10 workbook rows are
eligible.

Every run centers the ROI on the predicted pancreas; tumor-centered ROIs are not produced. Patients
without a pancreas prediction produce no ROI artifacts.

The preprocess step strictly skips missing/invalid and out-of-bounds ranges; the pipeline accepts
every curated range and only skips folders with no CT series or no requested mask. Its manifest
records the selected instance numbers, SOP Instance UIDs, and any geometry warnings. Each detected
case contains the full-volume pancreas mask, the pancreas-centered cropped CT and mask, physical
bounding-box metadata, and an axial/coronal/sagittal review montage. All masks and ROIs are
provisional research outputs and must not be used diagnostically.

## Image embeddings

The separate `pc-image-embed` pipeline consumes `roi_ct.nii.gz` artifacts from one completed image
ROI run. Select a backend with `--encoder spectre|merlin`. Both encoders require a pancreas-ROI run
(all ROI runs center on the pancreas) containing a non-empty `roi_pancreas_mask.nii.gz` for every
encoded case.
For example:

```powershell
uv run pc-image-embed run `
  --encoder spectre `
  --roi-run "outputs/image_roi/<pancreas-roi-run>"

uv run pc-image-embed run `
  --encoder merlin `
  --roi-run "outputs/image_roi/<pancreas-roi-run>"
```

The command dispatches to the same verified ROCm interpreter and has no silent CPU fallback. Use
`embed` instead of `run` to require an already cached checkpoint. `--patients`, `--run-dir`,
`--resume`, and `--force` are supported.

The backends are pinned and save float32 embeddings without L2 normalization:

- [SPECTRE-Large](https://github.com/cclaess/SPECTRE) preserves native spacing, centers a crop on
  the predicted pancreas bounding box with a 15 mm margin, pads to 128 x 128 x 64 grid multiples,
  and saves the 1,080-value scan CLS token. Its model weights are CC-BY-NC-SA and restricted to
  non-commercial use.
- [Merlin](https://github.com/StanfordMIMI/Merlin) reorients to RAS, resamples to
  1.5 x 1.5 x 3 mm, clips `[-1000, 1000]` HU into `[0, 1]`, centers a 224 x 224 x 160 input on the
  pancreas, loads only the I3ResNet image substate, and saves its 2,048-value pooled embedding.

The package versions, Hugging Face revisions, and checkpoint SHA-256 values are pinned in the
source and recorded in every run manifest. The launcher enumerates physical ROCm devices before
Torch import, exposes the RX 6900 XT through `HIP_VISIBLE_DEVICES`, and verifies it as logical
`cuda:0`.

Each run writes:

- `image_embeddings.npz`: aligned patient IDs, encoder name, and the fixed float32 matrix.
- `patch_embeddings.npz`: patch/scan token vectors, locations, and valid-voxel counts.
- `embedding_summary.csv`: one audit row per selected ROI.
- `run_manifest.json`: input/model hashes, preprocessing, pooling, runtime, and failures.

## Verification

```powershell
uv run ruff check .
uv run pytest
```

GPU-marked tests require the pinned AMD ROCm runtime and RX 6900 XT; they skip unless
`RUN_IMAGE_GPU_TEST=1` is set.
