"""Tests for CLI wiring, subcommands, and end-to-end synthetic dry run."""

import hashlib
import json
import shutil
from pathlib import Path
import pytest
from typer.testing import CliRunner

from mmsar.cli import app
from tests.fixtures.make_synthetic_video import create_synthetic_video
from mmsar.preprocessing.video_io import decimate_and_crop
from mmsar.fusion import late_fusion
from mmsar.postprocessing.baseline import BaselinePostProcessor
from mmsar.postprocessing.history_tracking import HistoryTrackingPostProcessor
from mmsar.postprocessing.moving_average import MovingAveragePostProcessor
from mmsar.postprocessing.median_filter import MedianFilterPostProcessor
from mmsar.postprocessing.mamba_sssm import MambaSSSMPostProcessor
from mmsar.metrics.frame_metrics import calculate_frame_metrics, count_false_alarms
from mmsar.metrics.alarm_metrics import calculate_alarm_metrics
from mmsar.metrics.report import to_markdown_table, TABLE_1B_COLUMNS
from mmsar.annotation.label_studio_config import generate_label_studio_config
from mmsar.training.data_yaml import generate_data_yaml
from mmsar.training.train import run_training
from mmsar.reporting.fill_readme_tables import fill_readme_tables


runner = CliRunner()
REPO_ROOT = Path(__file__).resolve().parent.parent
REAL_README = REPO_ROOT / "README.md"
REAL_DATA_RAW = REPO_ROOT / "data" / "raw"


SUBCOMMANDS = [
    "version",
    "status",
    "preprocess",
    "sample-frames",
    "push-label-studio",
    "import-annotations",
    "train",
    "fill-results",
]


def test_cli_root_help() -> None:
    """Ensure root CLI --help succeeds and lists all 8 subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, f"mmsar --help failed: {result.stdout}"
    for subcmd in SUBCOMMANDS:
        assert subcmd in result.stdout, f"Subcommand '{subcmd}' missing from mmsar --help output"


@pytest.mark.parametrize("subcmd", SUBCOMMANDS)
def test_cli_subcommand_help(subcmd: str) -> None:
    """Ensure every subcommand responds to --help with exit code 0."""
    result = runner.invoke(app, [subcmd, "--help"])
    assert result.exit_code == 0, f"mmsar {subcmd} --help failed: {result.stdout}"


def test_cli_version_command() -> None:
    """Test mmsar version outputs package version and exits 0."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.stdout


def test_cli_status_command() -> None:
    """Test mmsar status outputs manifest progress and exits 0."""
    result = runner.invoke(app, ["status"])
    assert result.exit_code == 0
    assert "desert/positive: 5/5" in result.stdout
    assert "desert/hard_negative: 3/5" in result.stdout


def test_synthetic_end_to_end_pipeline(tmp_path: Path) -> None:
    """Full-pipeline synthetic dry run:
    video -> decimate/crop -> late fusion -> 5 post-processors -> metrics -> report table
    -> annotation config -> data.yaml -> training dry run -> README results filler dry run.
    Must touch ZERO files in data/raw/ and ZERO bytes in README.md.
    """
    initial_readme_hash = hashlib.sha256(REAL_README.read_bytes()).hexdigest()
    raw_files_before = set(REAL_DATA_RAW.rglob("*"))

    # Step 1: Generate synthetic raw video in tmp_path
    synthetic_video = tmp_path / "synthetic_raw.mp4"
    create_synthetic_video(output_path=synthetic_video, duration_sec=2, fps=24)
    assert synthetic_video.exists() and synthetic_video.stat().st_size > 0

    # Step 2: Preprocess (decimate 3:1 & crop to 640x640)
    frames_dir = tmp_path / "processed_frames"
    extracted_frames = decimate_and_crop(video_path=synthetic_video, out_dir=frames_dir)
    assert len(extracted_frames) == 16, f"Expected 16 frames, got {len(extracted_frames)}"
    for f in extracted_frames:
        assert f.exists()

    # Step 3: Upstream confidence sequences & late fusion
    c_rgb = [0.1, 0.2, 0.8, 0.9, 0.85, 0.9, 0.1, 0.05, 0.7, 0.8]
    c_thermal = [0.05, 0.15, 0.75, 0.95, 0.8, 0.85, 0.05, 0.1, 0.65, 0.75]
    fused_scores = late_fusion(c_rgb, c_thermal)
    assert len(fused_scores) == 10
    assert all(isinstance(v, float) for v in fused_scores)

    # Step 4: Run all 5 post-processing methods
    tau = 0.5
    gt = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1]

    processors = {
        "Baseline": BaselinePostProcessor(),
        "5-Frame History Tracking": HistoryTrackingPostProcessor(window_size=5, required_hits=3),
        "5-Frame Moving Average": MovingAveragePostProcessor(window_size=5),
        "5-Frame Median Filter": MedianFilterPostProcessor(window_size=5),
    }

    # PyTorch Mamba-SSSM
    mamba = MambaSSSMPostProcessor(d_model=16, d_state=8)
    # Quick fit on synthetic pattern
    mamba.fit([[0.1, 0.2, 0.8, 0.9, 0.85, 0.9, 0.1, 0.05, 0.7, 0.8]], [gt], epochs=10, lr=0.01)
    processors["Mamba-SSSM"] = mamba

    table_rows = []
    fill_metrics: dict[str, str] = {}

    for name, proc in processors.items():
        decisions = proc.decide(fused_scores, tau=tau)
        assert len(decisions) == len(fused_scores)
        assert all(d in (0, 1) for d in decisions)

        # Step 5: Metrics computation
        frame_m = calculate_frame_metrics(y_true=gt, y_pred=decisions)
        alarm_m = calculate_alarm_metrics(y_true=gt, y_pred=decisions)
        fp, fr, ff1 = frame_m["precision"], frame_m["recall"], frame_m["f1"]
        ap, ar, af1 = alarm_m["precision"], alarm_m["recall"], alarm_m["f1"]
        fa_count = count_false_alarms(y_true=gt, y_pred=decisions)["total"]

        assert 0.0 <= fp <= 1.0
        assert 0.0 <= fr <= 1.0
        assert 0.0 <= ff1 <= 1.0
        assert 0.0 <= ap <= 1.0
        assert 0.0 <= ar <= 1.0
        assert 0.0 <= af1 <= 1.0
        assert fa_count >= 0

        table_rows.append([
            "YOLO11n",
            name,
            f"{fp:.3f}",
            f"{fr:.3f}",
            f"{ap:.3f}",
            f"{ar:.3f}",
        ])

        fill_metrics[f"{name}::frame precision"] = f"{fp:.3f}"
        fill_metrics[f"{name}::frame recall"] = f"{fr:.3f}"

    # Step 6: Render markdown results table
    markdown_table = to_markdown_table(rows=table_rows, columns=TABLE_1B_COLUMNS)
    assert "|" in markdown_table
    for col in TABLE_1B_COLUMNS:
        assert col in markdown_table

    # Step 7: Annotation XML config generation
    xml_config = generate_label_studio_config()
    assert "<View>" in xml_config
    assert "Person_Detected" in xml_config

    # Step 8: Training data.yaml generation
    data_yaml_path = tmp_path / "data.yaml"
    data_config = generate_data_yaml(
        manifest_source=REPO_ROOT / "data" / "manifest.json",
        output_path=data_yaml_path,
        dataset_root=tmp_path / "processed",
    )
    assert data_yaml_path.exists()
    assert data_config["names"][0] == "Person_Detected"

    # Step 9: Training dry-run validation
    train_result = run_training(
        model_arch="yolo11n.yaml",
        data_config=data_yaml_path,
        hyperparams_path=REPO_ROOT / "configs" / "hyperparams.yaml",
        dry_run=True,
    )
    assert train_result["status"] == "dry_run_success"
    assert train_result["batch"] == 32

    # Step 10: Reporting table filler dry-run and confirmed fill on isolated copy
    readme_copy = tmp_path / "README.md"
    shutil.copyfile(REAL_README, readme_copy)
    diff = fill_readme_tables(
        readme_path=readme_copy,
        metrics=fill_metrics,
        confirm=True,
        dry_run=False,
    )
    assert diff, "Expected non-empty diff for filled table copy"
    updated_copy_text = readme_copy.read_text(encoding="utf-8")
    for val in fill_metrics.values():
        assert val in updated_copy_text

    # Step 11: Assert zero side-effects on real repository files
    final_readme_hash = hashlib.sha256(REAL_README.read_bytes()).hexdigest()
    assert final_readme_hash == initial_readme_hash, "Real README.md was modified during test!"

    raw_files_after = set(REAL_DATA_RAW.rglob("*"))
    assert raw_files_after == raw_files_before, "data/raw was modified during test!"
