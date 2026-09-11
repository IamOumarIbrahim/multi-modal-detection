"""Loading and saving utilities for dataset manifest."""

import json
from pathlib import Path
from typing import Union
from mmsar.manifest.schema import Manifest


DEFAULT_MANIFEST_PATH = Path("data/manifest.json")


def load_manifest(path: Union[str, Path] = DEFAULT_MANIFEST_PATH) -> Manifest:
    """Load the dataset manifest from a JSON file.

    Args:
        path: Path to the manifest.json file.

    Returns:
        Manifest dataclass instance.
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Manifest file not found at {file_path}")

    data = json.loads(file_path.read_text(encoding="utf-8"))
    return Manifest.from_dict(data)


def save_manifest(
    manifest: Manifest,
    path: Union[str, Path] = DEFAULT_MANIFEST_PATH,
) -> None:
    """Save the dataset manifest to a JSON file with pretty indentation.

    Args:
        manifest: Manifest instance to serialize.
        path: Path to write the manifest.json file.
    """
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(manifest.to_dict(), indent=2) + "\n"
    file_path.write_text(content, encoding="utf-8")
