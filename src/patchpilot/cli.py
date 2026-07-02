"""Command-line interface for PatchPilot."""

import typer

app = typer.Typer(
    help="Bounded autonomous Python debugging and repair agent.",
    no_args_is_help=True,
)


@app.callback()
def main() -> None:
    """Run PatchPilot commands."""


@app.command()
def version() -> None:
    """Display the installed PatchPilot version."""
    from patchpilot import __version__

    typer.echo(__version__)


if __name__ == "__main__":
    app()
