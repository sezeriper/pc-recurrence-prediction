from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pc_recurrence.image_data.constants import DEFAULT_WORKBOOK

from .constants import DEFAULT_OUTPUT_ROOT
from .pipeline import (
    DEFAULT_SPLIT_SEED,
    DEFAULT_VALIDATION_FRACTION,
    preprocess_clinical_features,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Preprocess clinical variables and preserve CT report text for later embedding.",
)


@app.callback()
def main() -> None:
    """Run clinical-table preparation stages."""


def _selected_patients(value: str | None) -> set[str] | None:
    if not value:
        return None
    return {item.strip() for item in value.split(",") if item.strip()}


@app.command()
def preprocess(
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    patients: Annotated[
        str | None,
        typer.Option(help="Comma-separated workbook patient IDs or DICOM-folder aliases."),
    ] = None,
    fit_patients: Annotated[
        str | None,
        typer.Option(
            help=(
                "Optional comma-separated subset used to fit normalization and categorical "
                "vocabularies. Use the training split here to avoid leakage."
            )
        ),
    ] = None,
    validation_fraction: Annotated[
        float,
        typer.Option(
            help="Fraction of selected patients assigned to validation (default: 0.20)."
        ),
    ] = DEFAULT_VALIDATION_FRACTION,
    split_seed: Annotated[
        int,
        typer.Option(help="Non-negative seed for the deterministic patient split."),
    ] = DEFAULT_SPLIT_SEED,
) -> None:
    """Write separate normalized/raw train and validation clinical tables."""
    destination = preprocess_clinical_features(
        workbook,
        output_root,
        run_dir=run_dir,
        patients=_selected_patients(patients),
        fit_patients=_selected_patients(fit_patients),
        validation_fraction=validation_fraction,
        split_seed=split_seed,
        progress=typer.echo,
    )
    typer.echo(str(destination.resolve()))


if __name__ == "__main__":
    app()
