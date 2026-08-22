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

Source DICOM and the workbook are read from `dataset/`, which remains read-only. Generated review,
curated DICOM, model, and run artifacts default to `outputs/` or `.cache/`; no default command
writes inside `dataset/`.

Create an operator review inventory before curating any slices:

```powershell
uv run pc-image-roi inventory
```

This CPU-only command scans the untouched cleaned source recursively and writes
`outputs/ct_series_review/scan_selection.csv` plus axial montages under
`outputs/ct_series_review/previews/`. A DICOM Study groups an imaging encounter; a Series identifies
one acquisition within that Study. Review the Study/Series descriptions, `status`, concrete
`reason`, geometry warnings, and preview for every candidate.
`ready` means the classic single-frame CT Series can map the
workbook's one-based inclusive `Görüntü alanı` ordinals after ascending `InstanceNumber`.
`not_selectable` candidates remain visible for audit, including unsupported dose-report objects.
`no_series` identifies a missing folder or a folder without usable CT DICOM.

Open the loopback-only montage reviewer for that existing inventory:

```powershell
uv run pc-image-roi review
```

The reviewer keeps every patient and Series in inventory order, shows the original montage at full
resolution, and changes only `selected` cells when **Save selections** is pressed. It never chooses
a Series automatically. Partial progress and explicit clears are allowed; use `--selection FILE`
for another inventory, `--port` for a different loopback port, or `--no-open-browser` to print the
URL without launching a browser.

Choose exactly one `ready` row per patient in the reviewer, or set `selected=yes` directly in the
CSV and leave every other `selected` cell blank. Then curate:

```powershell
uv run pc-image-roi preprocess
```

Preprocessing validates the complete CSV against the live source before touching the curated
output under `outputs/dicom_selected`.
An added/removed Series, changed SOP membership, changed file counts, or changed workbook range
makes the selection stale and requires rerunning `inventory`. The chosen Study+Series is
deduplicated by SOP Instance UID only when duplicate files are byte-identical, the workbook range
is applied within that Series, and the exact resulting file set replaces any older curated range
atomically. Existing `scan_selection.csv` review work is protected unless `inventory --force` is
used; that flag deliberately resets selections and previews. Preprocessing `--force` stages and
rebuilds the exact selected output.

Both commands accept `--patients "Patient 1,PATIENT853534"` with workbook IDs or DICOM folder
aliases. Inventory and preprocessing then require choices only for that explicit subset. Use the
same subset for both commands.

`inspect` and segmentation consume only the curated one-Series patient directories. They never
choose a larger Series or repeat the operator's selection. Study UID, Series UID, selected SOP
Instance UIDs, and geometry warnings remain in inspection, segmentation, state, and bounding-box
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
requirement is unavailable.

Every run centers the ROI on the predicted pancreas; tumor-centered ROIs are not produced. Patients
without a pancreas prediction produce no ROI artifacts.

Preprocessing fails closed unless every targeted patient has one live `ready` choice. Geometry
gaps and duplicate slice positions are warnings rather than selection blockers. The segmentation
pipeline skips folders with zero or multiple processable CT Series instead of selecting one
implicitly. Each detected case contains the full-volume pancreas mask, the pancreas-centered
cropped CT and mask, physical bounding-box metadata, and an axial/coronal/sagittal review montage.
All masks and ROIs are provisional research outputs and must not be used diagnostically.

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
`--resume`, `--force`, and `--centering` are supported.

Both encoders center their crop on the CT volume's geometric center by default. Pass
`--centering pancreas` to center the crop on the predicted pancreas bounding box instead; window
sizes are unchanged. The chosen mode and the pancreas/volume center coordinates are recorded in
every run manifest and per-patient record.

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
