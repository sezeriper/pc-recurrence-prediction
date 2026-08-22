from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pc_recurrence.gpu_runtime import dispatch_to_gpu_runtime

from .artifacts import create_run_directory
from .constants import (
    DEFAULT_DICOM_ROOT,
    DEFAULT_MODEL_CACHE,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_RUNTIME_PYTHON,
    DEFAULT_SCAN_REVIEW_ROOT,
    DEFAULT_SCAN_SELECTION,
    DEFAULT_SOURCE_DICOM_ROOT,
    DEFAULT_WORKBOOK,
    EXPECTED_GPU_NAME,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Segment the pancreas/tumor and create review ROIs.",
)


def _selected_patients(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _dispatch_to_runtime(runtime_python: Path) -> None:
    try:
        dispatch_to_gpu_runtime(
            runtime_python,
            module="pc_recurrence.image_roi.cli",
            expected_gpu_name=EXPECTED_GPU_NAME,
        )
    except (RuntimeError, ValueError) as error:
        raise typer.BadParameter(str(error)) from error


@app.command()
def inspect(
    dicom_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated patient folder names.")
    ] = None,
    runtime_python: Annotated[Path, typer.Option()] = DEFAULT_RUNTIME_PYTHON,
) -> None:
    """Audit CT series geometry without running the segmentation model."""
    _dispatch_to_runtime(runtime_python)
    from .pipeline import inspect_dataset, write_inspection_run

    destination = run_dir or create_run_directory(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    inspections = inspect_dataset(dicom_root, _selected_patients(patients), workbook)
    write_inspection_run(inspections, destination, dicom_root, workbook)
    for item in inspections:
        typer.echo(f"{item.patient_id}: {item.geometry_status} ({item.reason or 'geometry valid'})")
    typer.echo(str(destination.resolve()))


@app.command()
def inventory(
    dicom_root: Annotated[
        Path, typer.Option(exists=True, file_okay=False)
    ] = DEFAULT_SOURCE_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_SCAN_REVIEW_ROOT,
    patients: Annotated[
        str | None,
        typer.Option(help="Comma-separated workbook patient IDs or DICOM folder names."),
    ] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Generate an editable CT Series inventory and axial review previews."""
    from .scan_selection import write_scan_inventory

    try:
        report = write_scan_inventory(
            dicom_root,
            workbook,
            output_dir,
            patients=_selected_patients(patients),
            force=force,
        )
    except (FileExistsError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    for summary in report.patient_summaries:
        issues = "; ".join(summary["issues"]) or "none"
        typer.echo(
            f"{summary['patient_id']}: {summary['candidate_count']} candidates, "
            f"{summary['ready_count']} ready (issues: {issues})"
        )
    typer.echo(str(output_dir.resolve()))


@app.command()
def review(
    selection: Annotated[
        Path,
        typer.Option(
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=True,
            help="Existing scan_selection.csv whose selected values will be edited.",
        ),
    ] = DEFAULT_SCAN_SELECTION,
    port: Annotated[
        int,
        typer.Option(
            min=1,
            max=65535,
            help="Loopback HTTP port for the local montage reviewer.",
        ),
    ] = 8765,
    open_browser: Annotated[
        bool,
        typer.Option(
            "--open-browser/--no-open-browser",
            help="Open the local montage reviewer in the default browser.",
        ),
    ] = True,
) -> None:
    """Review montages locally and edit selected values in an existing inventory CSV."""
    from .review_ui import serve_scan_review

    try:
        serve_scan_review(
            selection.resolve(),
            port=port,
            open_browser=open_browser,
        )
    except (OSError, RuntimeError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error


@app.command()
def preprocess(
    dicom_root: Annotated[
        Path, typer.Option(exists=True, file_okay=False)
    ] = DEFAULT_SOURCE_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_DICOM_ROOT,
    selection: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Inventory CSV containing one exact selected=yes Series per patient.",
        ),
    ] = DEFAULT_SCAN_SELECTION,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated workbook patient IDs or DICOM folder names.")
    ] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Curate exact operator-selected CT Series ranges for all downstream tasks."""
    from .preprocess import curate_dataset, write_curation_report

    try:
        report = curate_dataset(
            dicom_root,
            output_dir,
            workbook,
            selection,
            patients=_selected_patients(patients),
            force=force,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    write_curation_report(report)
    for patient in report.patients:
        outcome = patient.reason or "range copied"
        typer.echo(f"{patient.patient_id}: {patient.status} ({outcome})")
    if report.failures:
        raise typer.Exit(code=1)
    typer.echo(str(output_dir.resolve()))


def _run(
    dicom_root: Path,
    workbook: Path,
    output_root: Path,
    model_cache: Path,
    run_dir: Path | None,
    patients: str | None,
    resume: bool,
    force: bool,
    runtime_python: Path,
    *,
    local_model_only: bool,
) -> None:
    _dispatch_to_runtime(runtime_python)
    from .pipeline import run_segmentation

    destination = run_segmentation(
        dicom_root=dicom_root,
        workbook_path=workbook,
        output_root=output_root,
        model_cache=model_cache,
        run_dir=run_dir,
        patients=_selected_patients(patients),
        resume=resume,
        force=force,
        local_model_only=local_model_only,
    )
    typer.echo(str(destination.resolve()))


@app.command()
def segment(
    dicom_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    model_cache: Annotated[Path, typer.Option()] = DEFAULT_MODEL_CACHE,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated patient folder names.")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
    runtime_python: Annotated[Path, typer.Option()] = DEFAULT_RUNTIME_PYTHON,
) -> None:
    """Run from an already cached and checksum-verified model."""
    _run(
        dicom_root,
        workbook,
        output_root,
        model_cache,
        run_dir,
        patients,
        resume,
        force,
        runtime_python,
        local_model_only=True,
    )


@app.command(name="run")
def run_pipeline(
    dicom_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    model_cache: Annotated[Path, typer.Option()] = DEFAULT_MODEL_CACHE,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated patient folder names.")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
    runtime_python: Annotated[Path, typer.Option()] = DEFAULT_RUNTIME_PYTHON,
) -> None:
    """Acquire the pinned model, segment CT scans, and create review artifacts."""
    _run(
        dicom_root,
        workbook,
        output_root,
        model_cache,
        run_dir,
        patients,
        resume,
        force,
        runtime_python,
        local_model_only=False,
    )


if __name__ == "__main__":
    app()
