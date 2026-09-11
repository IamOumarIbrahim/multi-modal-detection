"""Ultralytics YOLO training runner with dry-run mode and CUDA OOM fallback."""

from pathlib import Path
from typing import Union, Optional, Any
import yaml
import torch
from ultralytics import YOLO


DEFAULT_HYPERPARAMS_PATH = Path("configs/hyperparams.yaml")
DEFAULT_DATA_CONFIG = Path("configs/data.template.yaml")


def load_hyperparams(path: Union[str, Path] = DEFAULT_HYPERPARAMS_PATH) -> dict[str, Any]:
    """Load hyperparameters from YAML config file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Hyperparameters file not found at {p}")
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_training(
    model_arch: str = "yolo11n.yaml",
    data_config: Union[str, Path] = DEFAULT_DATA_CONFIG,
    hyperparams_path: Union[str, Path] = DEFAULT_HYPERPARAMS_PATH,
    dry_run: bool = False,
    min_batch: int = 1,
) -> dict[str, Any]:
    """Instantiate and train a YOLO model with configuration validation and OOM fallback.

    Args:
        model_arch: Architecture YAML path/name (e.g. 'yolo11n.yaml' or 'yolo26n.yaml').
        data_config: Path to dataset YAML configuration.
        hyperparams_path: Path to hyperparams.yaml.
        dry_run: If True, only validates config and instantiates model without training loop.
        min_batch: Minimum allowable batch size for OOM halving fallback.

    Returns:
        Summary dict containing training status, model name, and final batch size.
    """
    hyperparams = load_hyperparams(hyperparams_path)
    batch = int(hyperparams.get("batch", 32))
    epochs = int(hyperparams.get("epochs", 100))
    patience = 0 if not hyperparams.get("early_stopping", False) else 50
    imgsz = int(hyperparams.get("imgsz", 640))

    # Instantiate model from architecture YAML (not .pt weights)
    model = YOLO(model_arch)

    if dry_run:
        return {
            "status": "dry_run_success",
            "model_arch": model_arch,
            "batch": batch,
            "epochs": epochs,
            "imgsz": imgsz,
        }

    # Training execution with automatic batch halving fallback on CUDA OOM
    current_batch = batch
    while current_batch >= min_batch:
        try:
            results = model.train(
                data=str(data_config),
                epochs=epochs,
                batch=current_batch,
                imgsz=imgsz,
                patience=patience,
                device=hyperparams.get("device", 0 if torch.cuda.is_available() else "cpu"),
            )
            return {
                "status": "train_success",
                "model_arch": model_arch,
                "final_batch": current_batch,
                "results": results,
            }
        except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
            if "out of memory" in str(e).lower():
                new_batch = current_batch // 2
                print(
                    f"CUDA OOM encountered at batch={current_batch}. "
                    f"Retrying with batch={new_batch}..."
                )
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                if new_batch < min_batch:
                    raise RuntimeError(
                        f"CUDA OOM persists and batch size {new_batch} is below minimum {min_batch}"
                    ) from e
                current_batch = new_batch
            else:
                raise
