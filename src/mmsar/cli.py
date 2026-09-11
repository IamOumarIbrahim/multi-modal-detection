"""MMSAR CLI Entrypoint."""

import json
from pathlib import Path
from typing import Optional
import typer
from mmsar import __version__
from mmsar.manifest.store import load_manifest, DEFAULT_MANIFEST_PATH
from mmsar.preprocessing.video_io import decimate_and_crop
from mmsar.preprocessing.frame_sampler import sample_frames_from_manifest
from mmsar.annotation.label_studio_config import generate_label_studio_config
from mmsar.annotation.label_studio_client import LabelStudioManager
from mmsar.annotation.rgb_to_thermal_copy import copy_annotations_rgb_to_thermal
from mmsar.training.train import run_training, DEFAULT_DATA_CONFIG, DEFAULT_HYPERPARAMS_PATH
from mmsar.reporting.fill_readme_tables import fill_readme_tables

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


@app.command(name="push-label-studio")
def push_label_studio(
    title: str = typer.Option("MMSAR Dataset", "--title", "-t", help="Label Studio project title."),
    url: str = typer.Option("http://localhost:8080", "--url", "-u", help="Label Studio URL."),
    api_key: str = typer.Option(..., "--api-key", "-k", help="Label Studio API key."),
    frames_dir: Optional[Path] = typer.Option(None, "--frames-dir", "-f", help="Directory of frames to upload."),
) -> None:
    """Create a project in Label Studio and optionally import frames."""
    manager = LabelStudioManager(base_url=url, api_key=api_key)
    config = generate_label_studio_config()
    project = manager.create_project(title=title, label_config=config)
    proj_id = getattr(project, "id", None)
    typer.echo(f"Created Label Studio project {proj_id}: {title}")

    if frames_dir and frames_dir.exists():
        image_files = sorted(frames_dir.glob("*.png")) + sorted(frames_dir.glob("*.jpg"))
        task_ids = manager.import_image_tasks(project_id=proj_id, image_paths=image_files)
        typer.echo(f"Imported {len(task_ids)} tasks into project {proj_id}")


@app.command(name="import-annotations")
def import_annotations(
    rgb_export: Path = typer.Option(..., "--rgb-export", "-r", help="Path to exported RGB annotations JSON."),
    mapping_file: Path = typer.Option(..., "--mapping", "-m", help="JSON file mapping RGB to Thermal frames."),
    out_thermal: Path = typer.Option(..., "--out", "-o", help="Path for generated Thermal annotations JSON."),
) -> None:
    """Copy RGB bounding-box annotations onto corresponding Thermal frames."""
    mapping = json.loads(mapping_file.read_text(encoding="utf-8"))
    thermal_tasks = copy_annotations_rgb_to_thermal(
        rgb_export_path=rgb_export,
        thermal_frame_ids=mapping,
        output_path=out_thermal,
    )
    typer.echo(f"Successfully converted {len(thermal_tasks)} annotations to {out_thermal}")


@app.command()
def train(
    model: str = typer.Option("yolo11n.yaml", "--model", help="YOLO architecture YAML (e.g. yolo11n.yaml, yolo26n.yaml)."),
    data: Path = typer.Option(DEFAULT_DATA_CONFIG, "--data", help="Dataset YAML configuration path."),
    hyperparams: Path = typer.Option(DEFAULT_HYPERPARAMS_PATH, "--hyperparams", help="Hyperparameters YAML path."),
    dry_run: bool = typer.Option(False, "--dry-run", flag_value=True, help="Dry run: validate config and instantiate model without training."),
) -> None:
    """Train YOLO detector or run a dry-run validation."""
    result = run_training(
        model_arch=model,
        data_config=data,
        hyperparams_path=hyperparams,
        dry_run=dry_run,
    )
    if dry_run:
        typer.echo(f"Dry run successful for model '{model}' with batch size {result['batch']}.")
    else:
        typer.echo(f"Training completed for model '{model}'.")


@app.command(name="fill-results")
def fill_results(
    metrics_file: Optional[Path] = typer.Option(None, "--metrics", "-m", help="Path to JSON metrics file."),
    readme: Path = typer.Option(Path("README.md"), "--readme", "-r", help="Path to README.md file."),
    dry_run: bool = typer.Option(False, "--dry-run", flag_value=True, help="Dry run: print diff without modifying README."),
    yes: bool = typer.Option(False, "--yes", "-y", flag_value=True, help="Explicit confirmation to write results to README.md."),
) -> None:
    """Fill TBD cells in README.md results tables from metrics file."""
    metrics_data = {}
    if metrics_file and metrics_file.exists():
        metrics_data = json.loads(metrics_file.read_text(encoding="utf-8"))

    if not dry_run and not yes:
        confirmed = typer.confirm("Are you sure you want to write results to README.md?")
        if not confirmed:
            typer.echo("Aborted by user.")
            raise typer.Abort()
        confirm_write = True
    else:
        confirm_write = yes

    diff = fill_readme_tables(
        readme_path=readme,
        metrics=metrics_data,
        confirm=confirm_write,
        dry_run=dry_run,
    )

    if dry_run:
        typer.echo("Dry run results diff:")
        typer.echo(diff if diff else "No changes proposed.")
    elif confirm_write:
        typer.echo(f"Successfully updated results in {readme}")


if __name__ == "__main__":
    app()
