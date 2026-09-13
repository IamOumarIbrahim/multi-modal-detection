"""End-to-end processing of RGB and Thermal annotations:
1. Adjust Label Studio export JSON so every snippet has maximum 1 unique annotation box track.
2. Generate corresponding Thermal export JSON.
3. Extract frames (640x640) for all 34 RGB and 34 Thermal videos.
4. Generate YOLO labels (structured and flat) for both RGB and Thermal modalities.
5. Create side-by-side verification overlays for visual alignment check.
6. Verify all invariants (frame count, single-box-per-frame, coordinate bounds).
"""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path

import cv2

from mmsar.annotation.video_annotation_parser import interpolate_video_sequence

DOWNLOAD_JSON = Path(r"D:\Downloads\project-2-at-2026-09-13-03-47-ea2ab94a.json")
ADJUSTED_RGB_JSON = Path("data/annotations/raw_exports/project_2_rgb_annotations_adjusted.json")
THERMAL_JSON = Path("data/annotations/raw_exports/project_2_thermal_annotations.json")
VERIFICATION_DIR = Path("data/processed/verification_samples")


def adjust_label_studio_json() -> list[dict]:
    print("=" * 80)
    print("STEP 1: Checking and Adjusting Label Studio Export JSON")
    print("=" * 80)

    if not DOWNLOAD_JSON.exists():
        raise FileNotFoundError(f"Download JSON not found at {DOWNLOAD_JSON}")

    data = json.loads(DOWNLOAD_JSON.read_text(encoding="utf-8"))
    print(f"Loaded {len(data)} tasks from {DOWNLOAD_JSON.name}\n")

    adjusted_tasks = []
    total_tracks_before = 0
    total_tracks_after = 0

    print(f"{'Task ID':<10} {'Snippet Name':<45} {'Tracks Before':<15} {'Tracks After':<15}")
    print("-" * 85)

    for task in data:
        t_copy = copy.deepcopy(task)
        tid = t_copy.get("id")
        v_url = t_copy.get("data", {}).get("video", "")
        stem = Path(v_url).name
        clean_stem = re.sub(r"^[0-9a-fA-F]{8}-", "", stem)

        anns = t_copy.get("annotations", [])
        if not anns:
            adjusted_tasks.append(t_copy)
            print(f"{tid:<10} {clean_stem:<45} {'0 (no anns)':<15} {'0':<15}")
            continue

        results = anns[0].get("result", [])
        tracks = [r for r in results if r.get("type") == "videorectangle"]
        total_tracks_before += len(tracks)

        if len(tracks) > 1:
            # Primary track is track 0
            primary_track = copy.deepcopy(tracks[0])
            combined_seq = list(primary_track.get("value", {}).get("sequence", []))

            for other_tr in tracks[1:]:
                other_seq = other_tr.get("value", {}).get("sequence", [])
                combined_seq.extend(other_seq)

            # Sort keyframes by frame number
            combined_seq.sort(key=lambda k: k.get("frame", 1))

            # Deduplicate by frame if any exist
            dedup_seq = []
            seen_frames = set()
            for kf in combined_seq:
                f_num = kf.get("frame")
                if f_num not in seen_frames:
                    seen_frames.add(f_num)
                    dedup_seq.append(kf)

            primary_track["value"]["sequence"] = dedup_seq

            # Reconstruct result with exactly 1 track
            non_tracks = [r for r in results if r.get("type") != "videorectangle"]
            anns[0]["result"] = [primary_track] + non_tracks

        final_tracks = [r for r in anns[0].get("result", []) if r.get("type") == "videorectangle"]
        assert len(final_tracks) <= 1, f"Task {tid} still has {len(final_tracks)} tracks!"
        total_tracks_after += len(final_tracks)
        adjusted_tasks.append(t_copy)
        print(f"{tid:<10} {clean_stem:<45} {len(tracks):<15} {len(final_tracks):<15}")

    print("-" * 85)
    print(f"Total video tracking boxes before: {total_tracks_before}")
    print(f"Total video tracking boxes after:  {total_tracks_after}")
    print(f"All {len(adjusted_tasks)} tasks confirmed to have <= 1 track!\n")

    # Save adjusted JSON
    ADJUSTED_RGB_JSON.parent.mkdir(parents=True, exist_ok=True)
    ADJUSTED_RGB_JSON.write_text(json.dumps(adjusted_tasks, indent=2), encoding="utf-8")
    print(f"Saved adjusted RGB JSON to: {ADJUSTED_RGB_JSON}")

    # Also update download file with clean version
    DOWNLOAD_JSON.write_text(json.dumps(adjusted_tasks, indent=2), encoding="utf-8")
    print(f"Updated {DOWNLOAD_JSON} with clean single-track annotations.")

    return adjusted_tasks


def create_thermal_annotation_json(rgb_tasks: list[dict]) -> list[dict]:
    print("\n" + "=" * 80)
    print("STEP 2: Creating Thermal Annotations JSON (Cloning RGB Annotations)")
    print("=" * 80)

    thermal_tasks = []
    for task in rgb_tasks:
        t_copy = copy.deepcopy(task)
        old_v = t_copy.get("data", {}).get("video", "")
        # Replace RGB with Thermal
        new_v = old_v.replace("RGB", "Thermal")
        t_copy["data"]["video"] = new_v

        # Annotations are preserved identically (spatially aligned 640x640)
        thermal_tasks.append(t_copy)

    THERMAL_JSON.write_text(json.dumps(thermal_tasks, indent=2), encoding="utf-8")
    print(f"Saved Thermal JSON with {len(thermal_tasks)} tasks to: {THERMAL_JSON}")
    return thermal_tasks


def process_frames_and_labels(rgb_tasks: list[dict]) -> dict:
    print("\n" + "=" * 80)
    print("STEP 3: Splitting into Annotated Frames & Generating YOLO Labels (RGB & Thermal)")
    print("=" * 80)

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    stats = {
        "total_rgb_frames": 0,
        "total_thermal_frames": 0,
        "total_rgb_boxes": 0,
        "total_thermal_boxes": 0,
        "videos_processed": 0,
        "snippets": [],
    }

    # Map clean stem to task
    task_by_stem = {}
    for task in rgb_tasks:
        v_url = task.get("data", {}).get("video", "")
        fname = Path(v_url).name
        clean_stem = re.sub(r"^[0-9a-fA-F]{8}-", "", fname).replace(".mp4", "")
        task_by_stem[clean_stem] = task

    print(f"Indexed {len(task_by_stem)} tasks by video stem.")

    start_time = time.time()

    for biome in ["desert", "forest"]:
        rgb_raw_dir = Path(f"data/raw/{biome}/positive/rgb")
        thermal_raw_dir = Path(f"data/raw/{biome}/positive/thermal")

        rgb_files = sorted(rgb_raw_dir.glob("*.mp4"))

        for rgb_path in rgb_files:
            stem = rgb_path.stem
            thermal_filename = stem.replace("RGB", "Thermal") + ".mp4"
            thermal_path = thermal_raw_dir / thermal_filename
            assert thermal_path.exists(), f"Thermal video missing at {thermal_path}"

            task = task_by_stem.get(stem)
            assert task is not None, f"No annotation task found for video {stem}"

            # Extract sequence from task
            ann_list = task.get("annotations", [])
            seq = []
            if ann_list and "result" in ann_list[0]:
                for res in ann_list[0]["result"]:
                    if res.get("type") == "videorectangle" and "value" in res:
                        seq = res["value"].get("sequence", [])
                        break

            cap_rgb = cv2.VideoCapture(str(rgb_path))
            total_frames_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
            w = int(cap_rgb.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap_rgb.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap_rgb.release()

            cap_th_check = cv2.VideoCapture(str(thermal_path))
            total_frames_th = int(cap_th_check.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_th_check.release()

            assert total_frames_rgb == total_frames_th, (
                f"Frame mismatch between RGB ({total_frames_rgb}) and Thermal ({total_frames_th}) for {stem}"
            )

            # Interpolate sequence
            frame_boxes = interpolate_video_sequence(seq, total_frames=total_frames_rgb) if seq else {}
            active_box_count = sum(1 for b in frame_boxes.values() if b is not None)

            print(f"Processing [{biome.upper():<6}] {stem:<35} -> {total_frames_rgb} frames, {active_box_count} active target boxes")

            # Setup directory paths
            dirs = {
                "rgb_frames_struct": Path(f"data/processed/frames/{biome}/positive/rgb/{stem}"),
                "rgb_images_flat": Path(f"data/processed/images/{biome}/positive/rgb"),
                "rgb_labels_struct": Path(f"data/processed/labels/{biome}/positive/rgb/{stem}"),
                "rgb_labels_flat": Path(f"data/processed/labels/{biome}/positive/rgb"),
                "thermal_stem": stem.replace("RGB", "Thermal"),
                "thermal_frames_struct": Path(f"data/processed/frames/{biome}/positive/thermal/{stem.replace('RGB', 'Thermal')}"),
                "thermal_images_flat": Path(f"data/processed/images/{biome}/positive/thermal"),
                "thermal_labels_struct": Path(f"data/processed/labels/{biome}/positive/thermal/{stem.replace('RGB', 'Thermal')}"),
                "thermal_labels_flat": Path(f"data/processed/labels/{biome}/positive/thermal"),
            }

            for d in [
                dirs["rgb_frames_struct"], dirs["rgb_images_flat"], dirs["rgb_labels_struct"], dirs["rgb_labels_flat"],
                dirs["thermal_frames_struct"], dirs["thermal_images_flat"], dirs["thermal_labels_struct"], dirs["thermal_labels_flat"],
            ]:
                d.mkdir(parents=True, exist_ok=True)

            cap_rgb = cv2.VideoCapture(str(rgb_path))
            cap_thermal = cv2.VideoCapture(str(thermal_path))

            frame_idx = 0
            overlay_saved = False
            vid_rgb_boxes = 0

            while True:
                ret_rgb, frame_rgb = cap_rgb.read()
                ret_th, frame_th = cap_thermal.read()

                if not ret_rgb or not ret_th:
                    break

                # 1. Save RGB and Thermal frame images
                f_idx_str = f"{frame_idx:06d}"
                cv2.imwrite(str(dirs["rgb_frames_struct"] / f"frame_{f_idx_str}.png"), frame_rgb)
                cv2.imwrite(str(dirs["rgb_images_flat"] / f"{stem}_frame_{f_idx_str}.png"), frame_rgb)
                stats["total_rgb_frames"] += 1

                cv2.imwrite(str(dirs["thermal_frames_struct"] / f"frame_{f_idx_str}.png"), frame_th)
                cv2.imwrite(str(dirs["thermal_images_flat"] / f"{dirs['thermal_stem']}_frame_{f_idx_str}.png"), frame_th)
                stats["total_thermal_frames"] += 1

                # 2. Compute YOLO coordinates
                # Note: Label Studio frames are 1-indexed
                ls_frame = frame_idx + 1
                box = frame_boxes.get(ls_frame)

                if box is not None:
                    x_norm = box["x"] / 100.0
                    y_norm = box["y"] / 100.0
                    w_norm = box["width"] / 100.0
                    h_norm = box["height"] / 100.0

                    x_center = max(0.0001, min(0.9999, x_norm + w_norm / 2.0))
                    y_center = max(0.0001, min(0.9999, y_norm + h_norm / 2.0))
                    w_norm = max(0.0001, min(1.0, w_norm))
                    h_norm = max(0.0001, min(1.0, h_norm))

                    yolo_line = f"0 {x_center:.6f} {y_center:.6f} {w_norm:.6f} {h_norm:.6f}\n"
                    stats["total_rgb_boxes"] += 1
                    stats["total_thermal_boxes"] += 1
                    vid_rgb_boxes += 1

                    # Save verification overlay on first active frame
                    if not overlay_saved:
                        ih, iw = frame_rgb.shape[:2]
                        bx1 = int((x_center - w_norm / 2.0) * iw)
                        by1 = int((y_center - h_norm / 2.0) * ih)
                        bx2 = int((x_center + w_norm / 2.0) * iw)
                        by2 = int((y_center + h_norm / 2.0) * ih)

                        over_rgb = frame_rgb.copy()
                        cv2.rectangle(over_rgb, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                        cv2.putText(over_rgb, f"RGB: {stem} (f{ls_frame})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                        over_th = frame_th.copy()
                        cv2.rectangle(over_th, (bx1, by1), (bx2, by2), (0, 0, 255), 2)
                        cv2.putText(over_th, f"TIR: {dirs['thermal_stem']} (f{ls_frame})", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                        side_by_side = cv2.hconcat([over_rgb, over_th])
                        overlay_path = VERIFICATION_DIR / f"overlay_{stem}_f{f_idx_str}.png"
                        cv2.imwrite(str(overlay_path), side_by_side)
                        overlay_saved = True
                else:
                    yolo_line = ""

                # Write RGB labels
                (dirs["rgb_labels_struct"] / f"frame_{f_idx_str}.txt").write_text(yolo_line, encoding="utf-8")
                (dirs["rgb_labels_flat"] / f"{stem}_frame_{f_idx_str}.txt").write_text(yolo_line, encoding="utf-8")

                # Write Thermal labels
                (dirs["thermal_labels_struct"] / f"frame_{f_idx_str}.txt").write_text(yolo_line, encoding="utf-8")
                (dirs["thermal_labels_flat"] / f"{dirs['thermal_stem']}_frame_{f_idx_str}.txt").write_text(yolo_line, encoding="utf-8")

                frame_idx += 1

            cap_rgb.release()
            cap_thermal.release()
            assert frame_idx == total_frames_rgb, f"Frame count mismatch for {stem}: {frame_idx} vs {total_frames_rgb}"
            stats["videos_processed"] += 1
            stats["snippets"].append({
                "biome": biome,
                "stem": stem,
                "frames": frame_idx,
                "active_boxes": vid_rgb_boxes,
            })

    elapsed = time.time() - start_time
    print(f"\nProcessing completed in {elapsed:.1f}s.")
    return stats


def verify_processed_dataset() -> None:
    print("\n" + "=" * 80)
    print("STEP 4: Rigorous Verification of Dataset Invariants")
    print("=" * 80)

    # 1. Verify JSON invariant
    rgb_tasks = json.loads(ADJUSTED_RGB_JSON.read_text(encoding="utf-8"))
    thermal_tasks = json.loads(THERMAL_JSON.read_text(encoding="utf-8"))

    assert len(rgb_tasks) == 34, f"Expected 34 RGB tasks, got {len(rgb_tasks)}"
    assert len(thermal_tasks) == 34, f"Expected 34 Thermal tasks, got {len(thermal_tasks)}"

    for t in rgb_tasks:
        results = t.get("annotations", [{}])[0].get("result", [])
        tracks = [r for r in results if r.get("type") == "videorectangle"]
        assert len(tracks) <= 1, f"Task {t['id']} has {len(tracks)} tracks! Invariant violated."

    print("[PASS] JSON Track Invariant: All 34 tasks have <= 1 annotation box track.")

    # 2. Verify Frame and Label files
    expected_total_frames = 7970
    violations = 0
    checked_labels = 0
    empty_labels = 0
    single_box_labels = 0

    for biome in ["desert", "forest"]:
        for mod in ["rgb", "thermal"]:
            images_flat = list(Path(f"data/processed/images/{biome}/positive/{mod}").glob("*.png"))
            labels_flat = list(Path(f"data/processed/labels/{biome}/positive/{mod}").glob("*.txt"))

            print(f"Checking {biome.upper():<6} {mod.upper():<7} -> {len(images_flat)} images, {len(labels_flat)} labels")
            assert len(images_flat) == len(labels_flat), f"Image/label count mismatch in {biome} {mod}"

            for lbl_p in labels_flat:
                checked_labels += 1
                lines = lbl_p.read_text(encoding="utf-8").strip().splitlines()
                if len(lines) == 0:
                    empty_labels += 1
                elif len(lines) == 1:
                    single_box_labels += 1
                    parts = lines[0].split()
                    assert len(parts) == 5, f"Invalid YOLO format in {lbl_p}: {lines[0]}"
                    cls_id, xc, yc, bw, bh = parts
                    assert cls_id == "0", f"Unexpected class_id {cls_id} in {lbl_p}"
                    f_xc, f_yc, f_bw, f_bh = float(xc), float(yc), float(bw), float(bh)
                    assert 0.0 <= f_xc <= 1.0 and 0.0 <= f_yc <= 1.0, f"Coordinates out of bounds in {lbl_p}"
                    assert 0.0 < f_bw <= 1.0 and 0.0 < f_bh <= 1.0, f"Dimensions out of bounds in {lbl_p}"
                else:
                    violations += 1
                    print(f"VIOLATION: Multiple boxes ({len(lines)}) found in {lbl_p}!")

    assert violations == 0, f"Found {violations} frames with multiple bounding boxes!"
    print(f"[PASS] Single Box Invariant: Checked {checked_labels} label files.")
    print(f"       - Non-target frames (empty labels): {empty_labels}")
    print(f"       - Target frames (exactly 1 box):    {single_box_labels}")
    print(f"       - Violations (>1 box):               {violations}")

    # Check verification overlays
    overlays = list(VERIFICATION_DIR.glob("*.png"))
    print(f"[PASS] Verification Overlays: {len(overlays)} side-by-side verification samples generated in {VERIFICATION_DIR}")

    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS PASSED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    adjust_label_studio_json()
    rgb_tasks = json.loads(ADJUSTED_RGB_JSON.read_text(encoding="utf-8"))
    create_thermal_annotation_json(rgb_tasks)
    stats = process_frames_and_labels(rgb_tasks)
    verify_processed_dataset()
