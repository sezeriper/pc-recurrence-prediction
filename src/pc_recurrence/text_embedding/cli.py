from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from .constants import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_MAX_LENGTH,
    DEFAULT_MODEL_CACHE,
    DEFAULT_OUTPUT_ROOT,
)

app = typer.Typer(
    no_args_is_help=True,
    help="Embed split CT reports with the pinned Qwen3-Embedding-8B model.",
)


def _run(
    clinical_run: Path,
    output_root: Path,
    model_cache: Path,
    run_dir: Path | None,
    batch_size: int,
    max_length: int,
    resume: bool,
    force: bool,
    skip_unavailable: bool,
    *,
    local_model_only: bool,
) -> None:
    from .pipeline import run_text_embedding

    destination = run_text_embedding(
        clinical_run,
        output_root,
        model_cache,
        run_dir=run_dir,
        batch_size=batch_size,
        max_length=max_length,
        resume=resume,
        force=force,
        skip_unavailable=skip_unavailable,
        local_model_only=local_model_only,
        progress=typer.echo,
    )
    typer.echo(str(destination.resolve()))


@app.command()
def embed(
    clinical_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    model_cache: Annotated[Path, typer.Option()] = DEFAULT_MODEL_CACHE,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option()] = DEFAULT_BATCH_SIZE,
    max_length: Annotated[int, typer.Option()] = DEFAULT_MAX_LENGTH,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
    skip_unavailable: Annotated[
        bool,
        typer.Option("--skip-unavailable/--require-all"),
    ] = True,
) -> None:
    """Embed using an already cached pinned Qwen snapshot."""
    _run(
        clinical_run,
        output_root,
        model_cache,
        run_dir,
        batch_size,
        max_length,
        resume,
        force,
        skip_unavailable,
        local_model_only=True,
    )


@app.command(name="run")
def run_pipeline(
    clinical_run: Annotated[Path, typer.Option(exists=True, file_okay=False)],
    output_root: Annotated[Path, typer.Option()] = DEFAULT_OUTPUT_ROOT,
    model_cache: Annotated[Path, typer.Option()] = DEFAULT_MODEL_CACHE,
    run_dir: Annotated[Path | None, typer.Option()] = None,
    batch_size: Annotated[int, typer.Option()] = DEFAULT_BATCH_SIZE,
    max_length: Annotated[int, typer.Option()] = DEFAULT_MAX_LENGTH,
    resume: Annotated[bool, typer.Option("--resume/--no-resume")] = True,
    force: Annotated[bool, typer.Option("--force")] = False,
    skip_unavailable: Annotated[
        bool,
        typer.Option("--skip-unavailable/--require-all"),
    ] = True,
) -> None:
    """Download the pinned Qwen snapshot if needed, then embed CT reports."""
    _run(
        clinical_run,
        output_root,
        model_cache,
        run_dir,
        batch_size,
        max_length,
        resume,
        force,
        skip_unavailable,
        local_model_only=False,
    )


if __name__ == "__main__":
    app()
