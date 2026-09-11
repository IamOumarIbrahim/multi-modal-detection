"""Generation of Ultralytics data.yaml from dataset manifest and annotation state."""

from pathlib import Path
from typing import Union, Optional, Any
import yaml
from mmsar.manifest.schema import Manifest
from mmsar.manifest.store import load_manifest, DEFAULT_MANIFEST_PATH


def generate_data_yaml(
    manifest_source: Optional[Union[Manifest, str, Path]] = None,
    output_path: Optional[Union[str, Path]] = None,
    dataset_root: Union[str, Path] = "data/processed",
) -> dict[str, Any]:
    """Generate an Ultralytics-compatible data.yaml configuration.

    Even when zero videos are currently annotated, produces a syntactically valid
    data.yaml with empty train/val/test splits without crashing.

    Args:
        manifest_source: Manifest instance or path to manifest.json (or None for default).
        output_path: Optional file path to write data.yaml.
        dataset_root: Root path for processed dataset frames.

    Returns:
        Dict representing data.yaml configuration.
    """
    if manifest_source is None:
        if DEFAULT_MANIFEST_PATH.exists():
            manifest = load_manifest(DEFAULT_MANIFEST_PATH)
        else:
            manifest = Manifest.create_empty()
    elif isinstance(manifest_source, (str, Path)):
        manifest = load_manifest(manifest_source)
    else:
        manifest = manifest_source

    root_path = Path(dataset_root).as_posix()

    data_config: dict[str, Any] = {
        "path": root_path,
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {
            0: "Person_Detected",
        },
    }

    if output_path is not None:
        out_file = Path(output_path)
        out_file.parent.mkdir(parents=True, exist_ok=True)
        with open(out_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(data_config, f, sort_keys=False)

    return data_config
