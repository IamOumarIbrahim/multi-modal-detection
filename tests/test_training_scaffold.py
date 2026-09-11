"""Tests for training scaffolding, data.yaml generation, and training dry-run."""

from pathlib import Path
import yaml
import pytest
from typer.testing import CliRunner
from mmsar.cli import app
from mmsar.manifest.schema import Manifest
from mmsar.training.data_yaml import generate_data_yaml
from mmsar.training.train import run_training, load_hyperparams


def test_data_yaml_generation_zero_annotated(tmp_path: Path) -> None:
    # Assert data_yaml.py produces valid YAML with zero annotated videos
    manifest = Manifest.create_empty()
    assert manifest.total_count == 0

    out_yaml = tmp_path / "data.yaml"
    config = generate_data_yaml(manifest_source=manifest, output_path=out_yaml)

    assert out_yaml.exists(), "data.yaml was not generated"

    # Verify parsed YAML is valid and contains standard Ultralytics keys
    parsed = yaml.safe_load(out_yaml.read_text(encoding="utf-8"))
    assert parsed["train"] == "images/train"
    assert parsed["val"] == "images/val"
    assert parsed["test"] == "images/test"
    assert 0 in parsed["names"] or "0" in parsed["names"]
    name_0 = parsed["names"].get(0) or parsed["names"].get("0")
    assert name_0 == "Person_Detected"


def test_train_dry_run_exits_zero_without_training(tmp_path: Path) -> None:
    # Assert train.py --dry-run exits 0 without launching a real multi-epoch loop
    result_11 = run_training(model_arch="yolo11n.yaml", dry_run=True)
    assert result_11["status"] == "dry_run_success"
    assert result_11["batch"] == 32
    assert result_11["model_arch"] == "yolo11n.yaml"

    result_26 = run_training(model_arch="yolo26n.yaml", dry_run=True)
    assert result_26["status"] == "dry_run_success"
    assert result_26["batch"] == 32
    assert result_26["model_arch"] == "yolo26n.yaml"

    # Test via CLI runner
    runner = CliRunner()
    cli_result = runner.invoke(app, ["train", "--dry-run"])
    assert cli_result.exit_code == 0
    assert "Dry run successful" in cli_result.output


def test_hyperparams_match_readme() -> None:
    hyperparams = load_hyperparams()
    assert hyperparams["batch"] == 32
    assert hyperparams["epochs"] == 100
    assert hyperparams["early_stopping"] is False
