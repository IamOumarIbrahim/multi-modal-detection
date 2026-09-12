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


def get_model_plan(
    include_optional: Optional[bool] = None,
    hyperparams_path: Union[str, Path] = DEFAULT_HYPERPARAMS_PATH,
) -> list[str]:
    """Determine list of model architectures to train.

    YOLO11n is always the primary model. YOLO26n is optional and only
    included if include_optional is True or if configured in hyperparams.yaml.

    Args:
        include_optional: Explicit override. If True, includes optional models.
                          If False, excludes optional models. If None, reads config.
        hyperparams_path: Path to hyperparams.yaml.

    Returns:
        List of model architecture filenames (e.g. ['yolo11n.yaml'] or ['yolo11n.yaml', 'yolo26n.yaml']).
    """
    try:
        hyperparams = load_hyperparams(hyperparams_path)
    except FileNotFoundError:
        hyperparams = {}

    primary = str(hyperparams.get("primary_model", "yolo11n.yaml"))
    optional_models = hyperparams.get("optional_models", ["yolo26n.yaml"])
    if isinstance(optional_models, str):
        optional_models = [optional_models]

    if include_optional is None:
        include = bool(hyperparams.get("include_optional", False))
    else:
        include = bool(include_optional)

    plan = [primary]
    if include:
        for opt in optional_models:
            if opt not in plan:
                plan.append(opt)

    return plan


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
    batch = int(hyperparams.get("batch", 16))
    epochs = int(hyperparams.get("epochs", 100))
    patience = 0 if not hyperparams.get("early_stopping", False) else int(hyperparams.get("patience", 20))
    imgsz = int(hyperparams.get("imgsz", 640))
    amp = bool(hyperparams.get("amp", False))

    # Extended benchmark hyperparameters
    optimizer = str(hyperparams.get("optimizer", "SGD"))
    lr0 = float(hyperparams.get("lr0", 0.01))
    lrf = float(hyperparams.get("lrf", 0.01))
    momentum = float(hyperparams.get("momentum", 0.937))
    weight_decay = float(hyperparams.get("weight_decay", 0.0005))
    warmup_epochs = int(hyperparams.get("warmup_epochs", 3))
    cos_lr = bool(hyperparams.get("cos_lr", True))
    mosaic = float(hyperparams.get("mosaic", 1.0))
    hsv_h = float(hyperparams.get("hsv_h", 0.015))
    hsv_s = float(hyperparams.get("hsv_s", 0.7))
    hsv_v = float(hyperparams.get("hsv_v", 0.4))
    fliplr = float(hyperparams.get("fliplr", 0.5))

    # Instantiate model from architecture YAML (not .pt weights)
    model = YOLO(model_arch)

    if dry_run:
        return {
            "status": "dry_run_success",
            "model_arch": model_arch,
            "batch": batch,
            "epochs": epochs,
            "imgsz": imgsz,
            "optimizer": optimizer,
            "lr0": lr0,
            "lrf": lrf,
            "early_stopping": bool(hyperparams.get("early_stopping", False)),
            "patience": patience,
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
                amp=amp,
                device=hyperparams.get("device", 0 if torch.cuda.is_available() else "cpu"),
                optimizer=optimizer,
                lr0=lr0,
                lrf=lrf,
                momentum=momentum,
                weight_decay=weight_decay,
                warmup_epochs=warmup_epochs,
                cos_lr=cos_lr,
                mosaic=mosaic,
                hsv_h=hsv_h,
                hsv_s=hsv_s,
                hsv_v=hsv_v,
                fliplr=fliplr,
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


def run_training_plan(
    model_list: Optional[list[str]] = None,
    include_optional: Optional[bool] = None,
    data_config: Union[str, Path] = DEFAULT_DATA_CONFIG,
    hyperparams_path: Union[str, Path] = DEFAULT_HYPERPARAMS_PATH,
    dry_run: bool = False,
    min_batch: int = 1,
) -> list[dict[str, Any]]:
    """Run training for a sequence of models specified by plan.

    Args:
        model_list: Optional explicit list of model architecture strings.
                    If omitted, resolved via get_model_plan().
        include_optional: Flag passed to get_model_plan if model_list is None.
        data_config: Path to dataset YAML configuration.
        hyperparams_path: Path to hyperparams.yaml.
        dry_run: If True, executes dry run validation.
        min_batch: Minimum allowable batch size.

    Returns:
        List of result summaries, one per model executed.
    """
    if model_list is None:
        model_list = get_model_plan(include_optional=include_optional, hyperparams_path=hyperparams_path)

    results: list[dict[str, Any]] = []
    for arch in model_list:
        res = run_training(
            model_arch=arch,
            data_config=data_config,
            hyperparams_path=hyperparams_path,
            dry_run=dry_run,
            min_batch=min_batch,
        )
        results.append(res)
    return results

