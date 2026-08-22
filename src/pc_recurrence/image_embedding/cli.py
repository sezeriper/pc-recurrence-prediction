from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .constants import (
    DEFAULT_MODEL_CACHE,
    DEFAULT_OUTPUT_ROOT,
    CenteringMode,
    ImageEncoderName,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Extract SPECTRE or Merlin embeddings from 3D CT ROIs.",
)


def _selected_patients(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


def _run(
    roi_run: Path,
    encoder: ImageEncoderName,
    output_root: Path,
    model_cache: Path,
    run_dir: Path | None,
    patients: str | None,
    centering: CenteringMode,
    resume: bool,
    force: bool,
    *,
    local_model_only: bool,
) -> None:
    from .pipeline import run_embedding

    destination = run_embedding(
        roi_run=roi_run,
        encoder_name=encoder,
        output_root=output_root,
        model_cache=model_cache,
        run_dir=run_dir,
        patients=_selected_patients(patients),
        centering=centering,
        resume=resume,
        force=force,
        local_model_only=local_model_only,
    )
    typer.echo(str(destination.resolve()))


@app.command()
def embed(
    roi_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    encoder: Annotated[ImageEncoderName, typer.Option()] = ImageEncoderName.SPECTRE,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    model_cache: Annotated[Path, typer.Option()] = DEFAULT_MODEL_CACHE,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated patient ROI directory names.")
    ] = None,
    centering: Annotated[
        CenteringMode,
        typer.Option(help="Crop centering: predicted pancreas bbox or CT volume center."),
    ] = CenteringMode.VOLUME,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Encode ROIs using an already cached and checksum-verified checkpoint."""
    _run(
        roi_run,
        encoder,
        output_root,
        model_cache,
        run_dir,
        patients,
        centering,
        resume,
        force,
        local_model_only=True,
    )


@app.command(name="run")
def run_pipeline(
    roi_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    encoder: Annotated[ImageEncoderName, typer.Option()] = ImageEncoderName.SPECTRE,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    model_cache: Annotated[Path, typer.Option()] = DEFAULT_MODEL_CACHE,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None, typer.Option(help="Comma-separated patient ROI directory names.")
    ] = None,
    centering: Annotated[
        CenteringMode,
        typer.Option(help="Crop centering: predicted pancreas bbox or CT volume center."),
    ] = CenteringMode.VOLUME,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
) -> None:
    """Acquire the selected pinned checkpoint and encode all selected ROIs."""
    _run(
        roi_run,
        encoder,
        output_root,
        model_cache,
        run_dir,
        patients,
        centering,
        resume,
        force,
        local_model_only=False,
    )


if __name__ == "__main__":
    app()
