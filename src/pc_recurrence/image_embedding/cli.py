from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pc_recurrence.image_data.constants import DEFAULT_DICOM_ROOT, DEFAULT_WORKBOOK

from .constants import DEFAULT_MODEL_CACHE, DEFAULT_OUTPUT_ROOT, ImageEncoderName

app = typer.Typer(
    no_args_is_help=True,
    help="Extract SPECTRE or Merlin embeddings from selected CT series ranges.",
)


def _selected_patients(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _run(
    dicom_root: Path,
    workbook: Path,
    encoder: ImageEncoderName,
    output_root: Path,
    model_cache: Path,
    run_dir: Path | None,
    patients: str | None,
    resume: bool,
    force: bool,
    skip_unavailable: bool,
    *,
    local_model_only: bool,
) -> None:
    from .pipeline import run_embedding

    destination = run_embedding(
        dicom_root=dicom_root,
        workbook_path=workbook,
        encoder_name=encoder,
        output_root=output_root,
        model_cache=model_cache,
        run_dir=run_dir,
        patients=_selected_patients(patients),
        resume=resume,
        force=force,
        local_model_only=local_model_only,
        skip_unavailable=skip_unavailable,
    )
    typer.echo(str(destination.resolve()))


@app.command()
def embed(
    dicom_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    encoder: Annotated[ImageEncoderName, typer.Option()] = ImageEncoderName.SPECTRE,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    model_cache: Annotated[Path, typer.Option()] = DEFAULT_MODEL_CACHE,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated workbook patient IDs or DICOM folders.")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
    skip_unavailable: Annotated[
        bool,
        typer.Option(
            "--skip-unavailable/--require-all",
            help="Skip missing or invalid curated CT series (default), or require every patient.",
        ),
    ] = True,
) -> None:
    """Encode selected CT ranges using an already cached and verified checkpoint."""
    _run(
        dicom_root,
        workbook,
        encoder,
        output_root,
        model_cache,
        run_dir,
        patients,
        resume,
        force,
        skip_unavailable,
        local_model_only=True,
    )


@app.command(name="run")
def run_pipeline(
    dicom_root: Annotated[Path, typer.Option(exists=True, file_okay=False)] = DEFAULT_DICOM_ROOT,
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    encoder: Annotated[ImageEncoderName, typer.Option()] = ImageEncoderName.SPECTRE,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    model_cache: Annotated[Path, typer.Option()] = DEFAULT_MODEL_CACHE,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated workbook patient IDs or DICOM folders.")
    ] = None,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
    skip_unavailable: Annotated[
        bool,
        typer.Option(
            "--skip-unavailable/--require-all",
            help="Skip missing or invalid curated CT series (default), or require every patient.",
        ),
    ] = True,
) -> None:
    """Acquire the selected pinned checkpoint and encode selected CT ranges."""
    _run(
        dicom_root,
        workbook,
        encoder,
        output_root,
        model_cache,
        run_dir,
        patients,
        resume,
        force,
        skip_unavailable,
        local_model_only=False,
    )


if __name__ == "__main__":
    app()
