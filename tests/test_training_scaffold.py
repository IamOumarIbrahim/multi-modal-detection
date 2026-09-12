"""Tests for training scaffolding, data.yaml generation, and training dry-run."""

from pathlib import Path
import yaml
import pytest
from typer.testing import CliRunner
from mmsar.cli import app
from mmsar.manifest.schema import Manifest
from mmsar.training.data_yaml import generate_data_yaml
from mmsar.training.train import run_training, run_training_plan, get_model_plan, load_hyperparams


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
    assert result_11["batch"] == 16
    assert result_11["model_arch"] == "yolo11n.yaml"

    result_26 = run_training(model_arch="yolo26n.yaml", dry_run=True)
    assert result_26["status"] == "dry_run_success"
    assert result_26["batch"] == 16
    assert result_26["model_arch"] == "yolo26n.yaml"

    # Test via CLI runner default (primary model only)
    runner = CliRunner()
    cli_result = runner.invoke(app, ["train", "--dry-run"])
    assert cli_result.exit_code == 0
    assert "model 'yolo11n.yaml'" in cli_result.output
    assert "model 'yolo26n.yaml'" not in cli_result.output


def test_model_plan_optional_handling() -> None:
    # By default, YOLO26n is optional and excluded
    plan_default = get_model_plan()
    assert plan_default == ["yolo11n.yaml"]

    # When include_optional is True, both models are included
    plan_all = get_model_plan(include_optional=True)
    assert plan_all == ["yolo11n.yaml", "yolo26n.yaml"]

    # When include_optional is False explicitly, only primary
    plan_primary = get_model_plan(include_optional=False)
    assert plan_primary == ["yolo11n.yaml"]


def test_cli_train_with_optional_models() -> None:
    runner = CliRunner()

    # With --include-optional, both models run in dry-run
    result = runner.invoke(app, ["train", "--include-optional", "--dry-run"])
    assert result.exit_code == 0
    assert "model 'yolo11n.yaml'" in result.output
    assert "model 'yolo26n.yaml'" in result.output

    # Explicit --model overrides plan
    result_custom = runner.invoke(app, ["train", "--model", "yolo26n.yaml", "--dry-run"])
    assert result_custom.exit_code == 0
    assert "model 'yolo26n.yaml'" in result_custom.output
    assert "model 'yolo11n.yaml'" not in result_custom.output


def test_hyperparams_match_readme() -> None:
    hyperparams = load_hyperparams()
    assert hyperparams["batch"] == 16
    assert hyperparams["epochs"] == 100
    assert hyperparams["early_stopping"] is True
    assert hyperparams["patience"] == 20
    assert hyperparams["primary_model"] == "yolo11n.yaml"
    assert "yolo26n.yaml" in hyperparams["optional_models"]
    assert hyperparams["include_optional"] is False
    assert hyperparams["optimizer"] == "SGD"
    assert hyperparams["momentum"] == 0.937
    assert hyperparams["weight_decay"] == 0.0005
    assert hyperparams["lr0"] == 0.01
    assert hyperparams["lrf"] == 0.01
    assert hyperparams["warmup_epochs"] == 3
    assert hyperparams["cos_lr"] is True
    assert hyperparams["mosaic"] == 1.0
    assert hyperparams["seeds"] == [0, 42, 1234]
    assert "arm_1_main_model" in hyperparams["experimental_arms"]
    assert "arm_2_architecture_baseline" in hyperparams["experimental_arms"]
    assert "arm_3_hard_negative_ablation" in hyperparams["experimental_arms"]
    assert "arm_4_tta" in hyperparams["experimental_arms"]

