"""MMSAR CLI Entrypoint."""

from pathlib import Path
from typing import Optional
import typer
from mmsar import __version__
from mmsar.manifest.store import load_manifest, DEFAULT_MANIFEST_PATH
from mmsar.preprocessing.video_io import decimate_and_crop
from mmsar.preprocessing.frame_sampler import sample_frames_from_manifest

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


@app.command()
def preprocess(
    video: Path = typer.Argument(..., help="Path to video file to decimate and crop."),
    out_dir: Path = typer.Option(
        Path("data/processed/frames"),
        "--out-dir",
        "-o",
        help="Directory to save decimated, cropped frames.",
    ),
) -> None:
    """Decimate (24 -> 8 fps) and crop (640x640) an individual video file."""
    frames = decimate_and_crop(video_path=video, out_dir=out_dir)
    typer.echo(f"Extracted {len(frames)} frames to {out_dir}")


@app.command(name="sample-frames")
def sample_frames(
    source: Path = typer.Option(..., "--source", "-s", help="Source directory containing raw videos."),
    dest: Path = typer.Option(..., "--dest", "-d", help="Destination directory for cropped frames."),
    manifest: Optional[Path] = typer.Option(None, "--manifest", "-m", help="Optional manifest path."),
) -> None:
    """Extract decimated and cropped frames across multiple raw videos."""
    summary = sample_frames_from_manifest(
        source_dir=source,
        dest_dir=dest,
        manifest_path=manifest,
    )
    typer.echo(f"Processed {summary['processed_videos']} videos, extracted {summary['total_frames']} frames to {dest}")


if __name__ == "__main__":
    app()
