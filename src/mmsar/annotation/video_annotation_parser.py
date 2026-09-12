"""Parser and interpolator for Label Studio video tracking annotations to YOLO format."""

import json
from pathlib import Path
from typing import Union, Optional, Any


def interpolate_video_sequence(
    sequence: list[dict[str, Any]],
    total_frames: int,
) -> dict[int, Optional[dict[str, float]]]:
    """Interpolate bounding box coordinates for each frame from Label Studio keyframes.

    Args:
        sequence: List of keyframe dicts containing 'frame', 'x', 'y', 'width', 'height', 'enabled'.
        total_frames: Total number of frames in the video (e.g. 80).

    Returns:
        Dict mapping frame_number (1-indexed) to bounding box dict or None if not enabled.
    """
    if not sequence:
        return {f: None for f in range(1, total_frames + 1)}

    # Sort keyframes by frame number
    sorted_kf = sorted(sequence, key=lambda k: k.get("frame", 1))

    frame_boxes: dict[int, Optional[dict[str, float]]] = {}

    for f in range(1, total_frames + 1):
        # If before first keyframe
        if f < sorted_kf[0]["frame"]:
            frame_boxes[f] = None
            continue

        # If at or after last keyframe
        if f >= sorted_kf[-1]["frame"]:
            last_kf = sorted_kf[-1]
            if last_kf.get("enabled", True):
                frame_boxes[f] = {
                    "x": float(last_kf["x"]),
                    "y": float(last_kf["y"]),
                    "width": float(last_kf["width"]),
                    "height": float(last_kf["height"]),
                }
            else:
                frame_boxes[f] = None
            continue

        # Find enclosing keyframes
        prev_kf = sorted_kf[0]
        next_kf = sorted_kf[-1]
        for i in range(len(sorted_kf) - 1):
            if sorted_kf[i]["frame"] <= f <= sorted_kf[i + 1]["frame"]:
                prev_kf = sorted_kf[i]
                next_kf = sorted_kf[i + 1]
                break

        if not prev_kf.get("enabled", True):
            frame_boxes[f] = None
            continue

        f0, f1 = prev_kf["frame"], next_kf["frame"]
        if f0 == f1:
            frame_boxes[f] = {
                "x": float(prev_kf["x"]),
                "y": float(prev_kf["y"]),
                "width": float(prev_kf["width"]),
                "height": float(prev_kf["height"]),
            }
        else:
            t = (f - f0) / float(f1 - f0)
            frame_boxes[f] = {
                "x": float(prev_kf["x"]) + t * (float(next_kf["x"]) - float(prev_kf["x"])),
                "y": float(prev_kf["y"]) + t * (float(next_kf["y"]) - float(prev_kf["y"])),
                "width": float(prev_kf["width"]) + t * (float(next_kf["width"]) - float(prev_kf["width"])),
                "height": float(prev_kf["height"]) + t * (float(next_kf["height"]) - float(prev_kf["height"])),
            }

    return frame_boxes


def convert_ls_video_export_to_yolo(
    export_json_path: Union[str, Path],
    output_labels_dir: Union[str, Path],
    total_frames_per_video: int = 80,
    decimation_ratio: int = 1,
    class_id: int = 0,
) -> dict[str, int]:
    """Convert exported Label Studio video tracking JSON to per-frame YOLO text annotations.

    Label Studio coordinates are percentages (0..100).
    YOLO expects: class_id x_center y_center width height (normalized 0..1).

    Args:
        export_json_path: Path to exported Label Studio JSON file.
        output_labels_dir: Directory to save per-frame .txt files.
        total_frames_per_video: Total frame count for each video (e.g. 240 for 24fps or 80 for 8fps).
        decimation_ratio: Sampling stride (e.g. 3 for 24->8 fps decimation, 1 for all frames).
        class_id: YOLO class ID (default 0 for Person_Detected).

    Returns:
        Dict summarizing total files written.
    """
    in_p = Path(export_json_path)
    out_p = Path(output_labels_dir)
    out_p.mkdir(parents=True, exist_ok=True)

    tasks = json.loads(in_p.read_text(encoding="utf-8"))
    if isinstance(tasks, dict):
        tasks = [tasks]

    total_written = 0

    for task in tasks:
        raw_name = task.get("file_upload", "") or task.get("data", {}).get("video", "video")
        video_stem = Path(raw_name).stem.replace("8fps_", "")

        annotations = task.get("annotations", [])
        if not annotations:
            continue

        results = annotations[0].get("result", [])
        sequences = []
        for res in results:
            val = res.get("value", {})
            if "sequence" in val:
                sequences.append(val["sequence"])

        if not sequences:
            continue

        frame_boxes = interpolate_video_sequence(sequences[0], total_frames=total_frames_per_video)

        if decimation_ratio > 1:
            # Stride sampling (e.g. 24 fps -> 8 fps: frames 1, 4, 7, ... -> frame_000000, frame_000001)
            selected_frames = list(range(1, total_frames_per_video + 1, decimation_ratio))
            for saved_idx, frame_num in enumerate(selected_frames):
                box = frame_boxes.get(frame_num)
                label_filename = out_p / f"{video_stem}_frame_{saved_idx:06d}.txt"
                if box is not None:
                    x_norm = box["x"] / 100.0
                    y_norm = box["y"] / 100.0
                    w_norm = box["width"] / 100.0
                    h_norm = box["height"] / 100.0

                    x_center = max(0.0, min(1.0, x_norm + w_norm / 2.0))
                    y_center = max(0.0, min(1.0, y_norm + h_norm / 2.0))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))

                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    label_filename.write_text(line, encoding="utf-8")
                else:
                    label_filename.write_text("", encoding="utf-8")
                total_written += 1
        else:
            for frame_num, box in frame_boxes.items():
                label_filename = out_p / f"{video_stem}_frame_{frame_num:06d}.txt"
                if box is not None:
                    x_norm = box["x"] / 100.0
                    y_norm = box["y"] / 100.0
                    w_norm = box["width"] / 100.0
                    h_norm = box["height"] / 100.0

                    x_center = max(0.0, min(1.0, x_norm + w_norm / 2.0))
                    y_center = max(0.0, min(1.0, y_norm + h_norm / 2.0))
                    w_norm = max(0.0, min(1.0, w_norm))
                    h_norm = max(0.0, min(1.0, h_norm))

                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    label_filename.write_text(line, encoding="utf-8")
                else:
                    label_filename.write_text("", encoding="utf-8")

                total_written += 1

    return {"total_labels_written": total_written}
