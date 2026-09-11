"""MMSAR CLI Entrypoint."""

from pathlib import Path
import typer
from mmsar import __version__
from mmsar.manifest.store import load_manifest, DEFAULT_MANIFEST_PATH

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


@app.command()
def status(
    manifest_path: Path = typer.Option(
        DEFAULT_MANIFEST_PATH,
        "--manifest",
        "-m",
        help="Path to dataset manifest JSON file.",
    ),
) -> None:
    """Display dataset collection and annotation progress across all 9 bins."""
    manifest = load_manifest(manifest_path)
    typer.echo("MMSAR Dataset Progress Manifest:")
    typer.echo("-" * 40)
    for cond, scenarios in manifest.bins.items():
        for scen, b in scenarios.items():
            status_str = f"{cond}/{scen}: {b.count}/{b.target}"
            typer.echo(status_str)
    typer.echo("-" * 40)
    typer.echo(f"Total: {manifest.total_count}/{manifest.total_target}")


if __name__ == "__main__":
    app()
