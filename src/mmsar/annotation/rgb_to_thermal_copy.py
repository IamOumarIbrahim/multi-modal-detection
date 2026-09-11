"""Copy bounding box annotations from RGB frames to corresponding Thermal frames."""

import json
import copy
from pathlib import Path
from typing import Union, Sequence, Mapping, Any, Optional


def copy_annotations_rgb_to_thermal(
    rgb_export_path: Union[str, Path],
    thermal_frame_ids: Union[Mapping[str, str], Sequence[str]],
    output_path: Optional[Union[str, Path]] = None,
) -> list[dict[str, Any]]:
    """Copy bounding box annotations from an RGB Label Studio export to Thermal frames.

    Coordinates (x, y, width, height) are preserved identically because RGB and
    Thermal camera frames are spatially aligned.

    Args:
        rgb_export_path: Path to Label Studio JSON export file containing RGB annotations.
        thermal_frame_ids: Either a dict mapping rgb_frame_id -> thermal_frame_id,
                           or an ordered list of thermal frame identifiers matching
                           tasks in rgb_export.
        output_path: Optional path to save the generated thermal export JSON.

    Returns:
        List of generated Label Studio task dicts for Thermal modality.
    """
    in_path = Path(rgb_export_path)
    if not in_path.exists():
        raise FileNotFoundError(f"RGB export file not found: {in_path}")

    tasks: list[dict[str, Any]] = json.loads(in_path.read_text(encoding="utf-8"))
    thermal_tasks: list[dict[str, Any]] = []

    is_mapping = isinstance(thermal_frame_ids, Mapping)

    for i, rgb_task in enumerate(tasks):
        # Determine matching thermal ID
        rgb_image = rgb_task.get("data", {}).get("image", "")
        rgb_id = str(rgb_task.get("id", i))

        if is_mapping:
            # Check either exact image path, basename, or task id
            thermal_id = (
                thermal_frame_ids.get(rgb_image)
                or thermal_frame_ids.get(Path(rgb_image).name)
                or thermal_frame_ids.get(rgb_id)
            )
            if thermal_id is None:
                continue
        else:
            if i >= len(thermal_frame_ids):
                break
            thermal_id = thermal_frame_ids[i]

        # Deepcopy task to ensure no mutation and identical box coordinates
        thermal_task = copy.deepcopy(rgb_task)
        thermal_task["id"] = i + 1000  # distinct synthetic id
        thermal_task["data"]["image"] = thermal_id

        # Retain all annotation results with exact coordinates
        thermal_tasks.append(thermal_task)

    if output_path is not None:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(thermal_tasks, indent=2) + "\n", encoding="utf-8")

    return thermal_tasks
