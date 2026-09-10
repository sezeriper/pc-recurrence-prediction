from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pc_recurrence.io import create_run_directory

from .constants import (
    DEFAULT_DICOM_ROOT,
    DEFAULT_INSPECTION_OUTPUT_ROOT,
    DEFAULT_SOURCE_DICOM_ROOT,
    DEFAULT_WORKBOOK,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Curate and inspect professionally selected CT slices.",
)


def _selected_patients(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


@app.command()
def inspect(
    dicom_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_INSPECTION_OUTPUT_ROOT,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated workbook patient IDs or DICOM folders.")
    ] = None,
) -> None:
    """Audit curated CT series geometry."""
    from .inspection import inspect_dataset, write_inspection_run

    destination = run_dir or create_run_directory(output_root)
    destination.mkdir(parents=True, exist_ok=True)
    inspections = inspect_dataset(dicom_root, _selected_patients(patients), workbook)
    write_inspection_run(inspections, destination, dicom_root, workbook)
    for item in inspections:
        typer.echo(f"{item.patient_id}: {item.geometry_status} ({item.reason or 'geometry valid'})")
    typer.echo(str(destination.resolve()))


@app.command()
def preprocess(
    dicom_root: Annotated[
        Path, typer.Option(exists=True, file_okay=False)
    ] = DEFAULT_SOURCE_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_dir: Annotated[Path, typer.Option()] = DEFAULT_DICOM_ROOT,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated workbook patient IDs or DICOM folder names.")
    ] = None,
    force: Annotated[bool, typer.Option("--force")] = False,
    skip_unavailable: Annotated[
        bool,
        typer.Option(
            "--skip-unavailable/--require-all",
            help="Skip missing or invalid CT folders (default), or require every patient.",
        ),
    ] = True,
) -> None:
    """Copy each professionally selected CT series into the curated dataset."""
    from .preprocess import curate_dataset, write_curation_report

    try:
        report = curate_dataset(
            dicom_root,
            output_dir,
            workbook,
            patients=_selected_patients(patients),
            force=force,
        )
    except (FileNotFoundError, ValueError) as error:
        typer.echo(str(error), err=True)
        raise typer.Exit(code=1) from error
    write_curation_report(report)
    for patient in report.patients:
        outcome = patient.reason or "source slices copied"
        typer.echo(f"{patient.patient_id}: {patient.status} ({outcome})")
    if report.failures and not skip_unavailable:
        raise typer.Exit(code=1)
    typer.echo(str(output_dir.resolve()))


if __name__ == "__main__":
    app()
