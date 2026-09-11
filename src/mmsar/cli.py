"""MMSAR CLI Entrypoint."""

import typer
from mmsar import __version__

app = typer.Typer(
    name="mmsar",
    help="MMSAR — Multi-Modal Search-And-Rescue Detection Pipeline CLI",
    add_completion=False,
)

@app.callback()
def main() -> None:
    """MMSAR root command group."""
    pass

@app.command()
def version() -> None:
    """Print the MMSAR package version."""
    typer.echo(f"mmsar version {__version__}")

if __name__ == "__main__":
    app()
