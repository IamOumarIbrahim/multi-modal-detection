"""Export and process Label Studio video annotations for Desert Positive snippets.

Ingests Label Studio export, reclassifies snippets lacking bounding boxes
into clear_negative, moves raw videos and frames accordingly, and generates
YOLO label files.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import requests

from mmsar.annotation.video_annotation_parser import interpolate_video_sequence

BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"
PROJECT_ID = 2
DEFAULT_EXPORT_PATH = Path(r"D:\Downloads\project-2-at-2026-09-12-12-00-141fe869.json")


def load_export_data(source_path: Path | None = None) -> list[dict[str, Any]]:
    """Load export data from a local file or download via Label Studio API."""
    raw_dir = Path("data/annotations/raw_exports")
    raw_dir.mkdir(parents=True, exist_ok=True)
    backup_file = raw_dir / "project_2_desert_positive_export.json"

    target_file = source_path or DEFAULT_EXPORT_PATH
    if target_file.exists():
        print(f"Reading export data from local file: {target_file}")
        data = json.loads(target_file.read_text(encoding="utf-8"))
        backup_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
        print(f"Saved raw export backup to: {backup_file}")
        return data

    print(f"Local export file not found at {target_file}. Attempting Label Studio API export...")
    session = requests.Session()
    session.get(f"{BASE_URL}/user/login/")
    csrf = session.cookies.get("csrftoken")
    session.post(
        f"{BASE_URL}/user/login/",
        data={"email": EMAIL, "password": PASSWORD, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{BASE_URL}/user/login/"},
    )

    export_resp = session.get(f"{BASE_URL}/api/projects/{PROJECT_ID}/export?export_type=JSON")
    assert export_resp.status_code == 200, f"Export failed: {export_resp.status_code}"
    tasks_data = export_resp.json()
    backup_file.write_text(json.dumps(tasks_data, indent=2), encoding="utf-8")
    print(f"Exported {len(tasks_data)} tasks from API and saved backup to: {backup_file}")
    return tasks_data


def cleanup_stale_files(video_stem: str, old_scenario: str) -> None:
    """Remove stale frame and label files if a video was previously placed in another scenario."""
    old_frames_dir = Path(f"data/processed/frames/desert/{old_scenario}/rgb/{video_stem}")
    if old_frames_dir.exists():
        shutil.rmtree(old_frames_dir)

    old_labels_dir = Path(f"data/processed/labels/desert/{old_scenario}/rgb/{video_stem}")
    if old_labels_dir.exists():
        shutil.rmtree(old_labels_dir)

    # Remove flat files
    old_images_base = Path(f"data/processed/images/desert/{old_scenario}/rgb")
    for f in old_images_base.glob(f"{video_stem}_frame_*.png"):
        f.unlink()

    old_labels_base = Path(f"data/processed/labels/desert/{old_scenario}/rgb")
    for f in old_labels_base.glob(f"{video_stem}_frame_*.txt"):
        f.unlink()


def process_annotations(tasks_data: list[dict[str, Any]]) -> dict[str, Any]:
    """Process tasks, classify scenarios, move raw videos, extract frames, and generate labels."""
    debug_samples_dir = Path("data/processed/verification_samples")
    debug_samples_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "positive_videos": [],
        "reclassified_clear_negative_videos": [],
        "total_positive_frames": 0,
        "total_clear_negative_frames": 0,
        "total_positive_boxes": 0,
    }

    print("\n=== STEP: Processing Tasks, Reclassifying, and Extracting Frames ===")

    for task in tasks_data:
        raw_video_url = task.get("data", {}).get("video", "")
        stem_raw = Path(raw_video_url).stem
        video_stem = re.sub(r"^[0-9a-fA-F]{8}-", "", stem_raw).replace("8fps_", "")

        # Extract sequence from annotations
        ann_list = task.get("annotations", [])
        seq: list[dict[str, Any]] = []
        if ann_list and "result" in ann_list[0]:
            for res in ann_list[0]["result"]:
                if res.get("type") == "videorectangle" and "value" in res:
                    seq = res["value"].get("sequence", [])
                    break

        has_bboxes = len(seq) > 0
        scenario = "positive" if has_bboxes else "clear_negative"

        # Determine raw video paths
        pos_raw = Path("data/raw/desert/positive/rgb") / f"{video_stem}.mp4"
        clr_raw = Path("data/raw/desert/clear_negative/rgb") / f"{video_stem}.mp4"

        if scenario == "clear_negative":
            summary["reclassified_clear_negative_videos"].append(video_stem)
            print(f"\n[RECLASSIFIED -> CLEAR NEGATIVE] Video: {video_stem} (0 keyframes)")
            clr_raw.parent.mkdir(parents=True, exist_ok=True)
            if pos_raw.exists() and not clr_raw.exists():
                shutil.move(str(pos_raw), str(clr_raw))
                print(f"  Moved raw video: {pos_raw} -> {clr_raw}")
            raw_video_path = clr_raw
            cleanup_stale_files(video_stem, old_scenario="positive")
        else:
            summary["positive_videos"].append(video_stem)
            print(f"\n[POSITIVE DETECTIONS] Video: {video_stem} ({len(seq)} keyframes)")
            pos_raw.parent.mkdir(parents=True, exist_ok=True)
            if clr_raw.exists() and not pos_raw.exists():
                shutil.move(str(clr_raw), str(pos_raw))
                print(f"  Moved raw video: {clr_raw} -> {pos_raw}")
            raw_video_path = pos_raw
            cleanup_stale_files(video_stem, old_scenario="clear_negative")

        assert raw_video_path.exists(), f"Raw video file missing at {raw_video_path}"

        # Target directories
        frames_dir = Path(f"data/processed/frames/desert/{scenario}/rgb/{video_stem}")
        images_flat_dir = Path(f"data/processed/images/desert/{scenario}/rgb")
        labels_struct_dir = Path(f"data/processed/labels/desert/{scenario}/rgb/{video_stem}")
        labels_flat_dir = Path(f"data/processed/labels/desert/{scenario}/rgb")

        frames_dir.mkdir(parents=True, exist_ok=True)
        images_flat_dir.mkdir(parents=True, exist_ok=True)
        labels_struct_dir.mkdir(parents=True, exist_ok=True)
        labels_flat_dir.mkdir(parents=True, exist_ok=True)

        # Frame interpolation
        frame_boxes = interpolate_video_sequence(seq, total_frames=240) if has_bboxes else {}

        cap = cv2.VideoCapture(str(raw_video_path))
        frame_idx = 0
        boxes_in_video = 0
        sample_saved = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 1. Save frame images (640x640 PNG)
            frame_path = frames_dir / f"frame_{frame_idx:06d}.png"
            flat_image_path = images_flat_dir / f"{video_stem}_frame_{frame_idx:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            cv2.imwrite(str(flat_image_path), frame)

            # 2. Labels
            struct_label_file = labels_struct_dir / f"frame_{frame_idx:06d}.txt"
            flat_label_file = labels_flat_dir / f"{video_stem}_frame_{frame_idx:06d}.txt"

            ls_frame_num = frame_idx + 1
            box = frame_boxes.get(ls_frame_num)

            if box is not None:
                x_norm = box["x"] / 100.0
                y_norm = box["y"] / 100.0
                w_norm = box["width"] / 100.0
                h_norm = box["height"] / 100.0

                x_center = max(0.0, min(1.0, x_norm + w_norm / 2.0))
                y_center = max(0.0, min(1.0, y_norm + h_norm / 2.0))
                w_norm = max(0.0, min(1.0, w_norm))
                h_norm = max(0.0, min(1.0, h_norm))

                line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                struct_label_file.write_text(line, encoding="utf-8")
                flat_label_file.write_text(line, encoding="utf-8")
                boxes_in_video += 1
                summary["total_positive_boxes"] += 1

                if not sample_saved:
                    overlay = frame.copy()
                    ih, iw = frame.shape[:2]
                    bx1 = int((x_center - w_norm / 2.0) * iw)
                    by1 = int((y_center - h_norm / 2.0) * ih)
                    bx2 = int((x_center + w_norm / 2.0) * iw)
                    by2 = int((y_center + h_norm / 2.0) * ih)
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(
                        overlay,
                        f"Person {w_norm*iw:.0f}x{h_norm*ih:.0f}px",
                        (bx1, max(15, by1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    sample_path = debug_samples_dir / f"overlay_{video_stem}_frame_{frame_idx:06d}.png"
                    cv2.imwrite(str(sample_path), overlay)
                    sample_saved = True
            else:
                struct_label_file.write_text("", encoding="utf-8")
                flat_label_file.write_text("", encoding="utf-8")

            if scenario == "positive":
                summary["total_positive_frames"] += 1
            else:
                summary["total_clear_negative_frames"] += 1

            frame_idx += 1

        cap.release()
        print(f"  Extracted {frame_idx} frames. Bboxes: {boxes_in_video}. Saved to {frames_dir}")

    return summary


def main() -> None:
    source = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    tasks_data = load_export_data(source)
    print(f"Loaded {len(tasks_data)} tasks.")
    summary = process_annotations(tasks_data)

    print("\n==================================================")
    print(" PROCESSING COMPLETE SUMMARY")
    print("==================================================")
    print(f"True Positive Snippets ({len(summary['positive_videos'])}):")
    for v in summary["positive_videos"]:
        print(f"  + {v}")
    print(f"Reclassified Clear Negative Snippets ({len(summary['reclassified_clear_negative_videos'])}):")
    for v in summary["reclassified_clear_negative_videos"]:
        print(f"  - {v}")
    print(f"Total Positive Frames:        {summary['total_positive_frames']} (expected {len(summary['positive_videos']) * 240})")
    print(f"Total Clear Negative Frames:  {summary['total_clear_negative_frames']} (expected {len(summary['reclassified_clear_negative_videos']) * 240})")
    print(f"Total Positive Bounding Boxes: {summary['total_positive_boxes']}")
    print("==================================================")


if __name__ == "__main__":
    main()


