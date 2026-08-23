from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pc_recurrence.image_data.constants import DEFAULT_WORKBOOK

from .constants import (
    DECISION_THRESHOLD,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_PREDICTION_OUTPUT_ROOT,
    TEST_FRACTION,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Train and apply recurrence heads over image embeddings.",
)


@app.command()
def train(
    merlin_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    spectre_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    workbook: Annotated[Path, typer.Option(exists=True, dir_okay=False)] = DEFAULT_WORKBOOK,
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    test_fraction: Annotated[float, typer.Option()] = TEST_FRACTION,
    seed_start: Annotated[int, typer.Option(min=0)] = 0,
    threshold: Annotated[float, typer.Option()] = DECISION_THRESHOLD,
) -> None:
    """Train independent Merlin and SPECTRE recurrence heads."""
    from .pipeline import train_recurrence_models

    destination = train_recurrence_models(
        merlin_run=merlin_run,
        spectre_run=spectre_run,
        workbook_path=workbook,
        output_root=output_root,
        run_dir=run_dir,
        test_fraction=test_fraction,
        seed_start=seed_start,
        threshold=threshold,
    )
    typer.echo(str(destination.resolve()))


@app.command()
def predict(
    model_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    merlin_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    spectre_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option()] = DEFAULT_PREDICTION_OUTPUT_ROOT,
    run_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Apply trained recurrence heads to independently validated embedding runs."""
    from .pipeline import predict_recurrence

    destination = predict_recurrence(
        model_run=model_run,
        merlin_run=merlin_run,
        spectre_run=spectre_run,
        output_root=output_root,
        run_dir=run_dir,
    )
    typer.echo(str(destination.resolve()))


if __name__ == "__main__":
    app()
