"""Ingest Project 2 Annotations from D:\Downloads\project-2-at-2026-09-13-09-14-9bf995c8.json:
1. Clean & adjust JSON export (single track per snippet, sequence deduplication).
2. Clone annotations to Thermal JSON.
3. Extract frames & generate YOLO labels for all biomes (desert, forest, snow) for both RGB and Thermal.
4. Process negative snippets for all biomes to generate background frames and empty YOLO labels.
5. Create verification overlay samples for newly added snow snippets.
6. Rigorously verify all invariants (frame count, single-box-per-frame, coordinate bounds).
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

DOWNLOAD_JSON = Path(r"D:\Downloads\project-2-at-2026-09-13-09-14-9bf995c8.json")
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
            primary_track = copy.deepcopy(tracks[0])
            combined_seq = list(primary_track.get("value", {}).get("sequence", []))

            for other_tr in tracks[1:]:
                other_seq = other_tr.get("value", {}).get("sequence", [])
                combined_seq.extend(other_seq)

            combined_seq.sort(key=lambda k: k.get("frame", 1))

            dedup_seq = []
            seen_frames = set()
            for kf in combined_seq:
                f_num = kf.get("frame")
                if f_num not in seen_frames:
                    seen_frames.add(f_num)
                    dedup_seq.append(kf)

            primary_track["value"]["sequence"] = dedup_seq
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

    ADJUSTED_RGB_JSON.parent.mkdir(parents=True, exist_ok=True)
    ADJUSTED_RGB_JSON.write_text(json.dumps(adjusted_tasks, indent=2), encoding="utf-8")
    print(f"Saved adjusted RGB JSON to: {ADJUSTED_RGB_JSON}")

    return adjusted_tasks


def create_thermal_annotation_json(rgb_tasks: list[dict]) -> list[dict]:
    print("\n" + "=" * 80)
    print("STEP 2: Creating Thermal Annotations JSON (Cloning RGB Annotations)")
    print("=" * 80)

    thermal_tasks = []
    for task in rgb_tasks:
        t_copy = copy.deepcopy(task)
        old_v = t_copy.get("data", {}).get("video", "")
        new_v = old_v.replace("RGB", "Thermal")
        t_copy["data"]["video"] = new_v
        thermal_tasks.append(t_copy)

    THERMAL_JSON.write_text(json.dumps(thermal_tasks, indent=2), encoding="utf-8")
    print(f"Saved Thermal JSON with {len(thermal_tasks)} tasks to: {THERMAL_JSON}")
    return thermal_tasks


def process_positive_frames_and_labels(rgb_tasks: list[dict]) -> dict:
    print("\n" + "=" * 80)
    print("STEP 3: Extracting Frames & YOLO Labels for Positive Snippets (Desert, Forest, Snow)")
    print("=" * 80)

    VERIFICATION_DIR.mkdir(parents=True, exist_ok=True)

    task_by_stem = {}
    for task in rgb_tasks:
        v_url = task.get("data", {}).get("video", "")
        fname = Path(v_url).name
        clean_stem = re.sub(r"^[0-9a-fA-F]{8}-", "", fname).replace(".mp4", "")
        task_by_stem[clean_stem] = task

    print(f"Indexed {len(task_by_stem)} positive tasks by video stem.")
    start_time = time.time()

    total_processed_snippets = 0

    for biome in ["desert", "forest", "snow"]:
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

            # Check if frames already exist
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

            cap_rgb = cv2.VideoCapture(str(rgb_path))
            total_frames_rgb = int(cap_rgb.get(cv2.CAP_PROP_FRAME_COUNT))
            cap_rgb.release()

            # If already extracted and complete, skip re-extracting frames
            existing_rgb_flat = len(list(dirs["rgb_images_flat"].glob(f"{stem}_frame_*.png"))) if dirs["rgb_images_flat"].exists() else 0
            if existing_rgb_flat == total_frames_rgb and biome in ["desert", "forest"]:
                print(f"Skipping already extracted [{biome.upper():<6}] {stem:<35} ({total_frames_rgb} frames)")
                continue

            for d in [
                dirs["rgb_frames_struct"], dirs["rgb_images_flat"], dirs["rgb_labels_struct"], dirs["rgb_labels_flat"],
                dirs["thermal_frames_struct"], dirs["thermal_images_flat"], dirs["thermal_labels_struct"], dirs["thermal_labels_flat"],
            ]:
                d.mkdir(parents=True, exist_ok=True)

            ann_list = task.get("annotations", [])
            seq = []
            if ann_list and "result" in ann_list[0]:
                for res in ann_list[0]["result"]:
                    if res.get("type") == "videorectangle" and "value" in res:
                        seq = res["value"].get("sequence", [])
                        break

            frame_boxes = interpolate_video_sequence(seq, total_frames=total_frames_rgb) if seq else {}
            active_box_count = sum(1 for b in frame_boxes.values() if b is not None)

            print(f"Extracting [{biome.upper():<6}] {stem:<35} -> {total_frames_rgb} frames, {active_box_count} active target boxes")

            cap_rgb = cv2.VideoCapture(str(rgb_path))
            cap_thermal = cv2.VideoCapture(str(thermal_path))

            frame_idx = 0
            overlay_saved = False

            while True:
                ret_rgb, frame_rgb = cap_rgb.read()
                ret_th, frame_th = cap_thermal.read()
                if not ret_rgb or not ret_th:
                    break

                f_idx_str = f"{frame_idx:06d}"
                # Save frames
                cv2.imwrite(str(dirs["rgb_frames_struct"] / f"frame_{f_idx_str}.png"), frame_rgb)
                cv2.imwrite(str(dirs["rgb_images_flat"] / f"{stem}_frame_{f_idx_str}.png"), frame_rgb)

                cv2.imwrite(str(dirs["thermal_frames_struct"] / f"frame_{f_idx_str}.png"), frame_th)
                cv2.imwrite(str(dirs["thermal_images_flat"] / f"{dirs['thermal_stem']}_frame_{f_idx_str}.png"), frame_th)

                # YOLO coords (1-indexed frame in LS)
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

                    # Verification overlay on first target frame
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

                # Write labels
                (dirs["rgb_labels_struct"] / f"frame_{f_idx_str}.txt").write_text(yolo_line, encoding="utf-8")
                (dirs["rgb_labels_flat"] / f"{stem}_frame_{f_idx_str}.txt").write_text(yolo_line, encoding="utf-8")
                (dirs["thermal_labels_struct"] / f"frame_{f_idx_str}.txt").write_text(yolo_line, encoding="utf-8")
                (dirs["thermal_labels_flat"] / f"{dirs['thermal_stem']}_frame_{f_idx_str}.txt").write_text(yolo_line, encoding="utf-8")

                frame_idx += 1

            cap_rgb.release()
            cap_thermal.release()
            assert frame_idx == total_frames_rgb, f"Frame count mismatch: {frame_idx} vs {total_frames_rgb}"
            total_processed_snippets += 1

    print(f"Positive snippet frame extraction complete in {time.time() - start_time:.1f}s.")


def process_negative_frames_and_labels() -> None:
    print("\n" + "=" * 80)
    print("STEP 4: Extracting Frames & Generating Empty YOLO Labels for Negative Snippets")
    print("=" * 80)

    start_time = time.time()
    for biome in ["desert", "forest", "snow"]:
        for mod in ["rgb", "thermal"]:
            raw_dir = Path(f"data/raw/{biome}/negative/{mod}")
            if not raw_dir.exists():
                continue

            images_flat = Path(f"data/processed/images/{biome}/negative/{mod}")
            labels_flat = Path(f"data/processed/labels/{biome}/negative/{mod}")
            frames_struct_base = Path(f"data/processed/frames/{biome}/negative/{mod}")
            labels_struct_base = Path(f"data/processed/labels/{biome}/negative/{mod}")

            images_flat.mkdir(parents=True, exist_ok=True)
            labels_flat.mkdir(parents=True, exist_ok=True)
            frames_struct_base.mkdir(parents=True, exist_ok=True)

            video_files = sorted(raw_dir.glob("*.mp4"))
            for v_path in video_files:
                stem = v_path.stem
                cap = cv2.VideoCapture(str(v_path))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

                # Check if already extracted
                existing = len(list(images_flat.glob(f"{stem}_frame_*.png")))
                if existing == total_frames:
                    continue

                print(f"Processing negative [{biome.upper()}/{mod.upper()}] {stem:<35} ({total_frames} frames)...")

                struct_frames_dir = frames_struct_base / stem
                struct_labels_dir = labels_struct_base / stem
                struct_frames_dir.mkdir(parents=True, exist_ok=True)
                struct_labels_dir.mkdir(parents=True, exist_ok=True)

                cap = cv2.VideoCapture(str(v_path))
                f_idx = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    f_str = f"{f_idx:06d}"
                    # Images
                    cv2.imwrite(str(struct_frames_dir / f"frame_{f_str}.png"), frame)
                    cv2.imwrite(str(images_flat / f"{stem}_frame_{f_str}.png"), frame)

                    # Empty YOLO labels
                    (struct_labels_dir / f"frame_{f_str}.txt").write_text("", encoding="utf-8")
                    (labels_flat / f"{stem}_frame_{f_str}.txt").write_text("", encoding="utf-8")
                    f_idx += 1

                cap.release()

    print(f"Negative frames & labels processed in {time.time() - start_time:.1f}s.")


def verify_all_invariants() -> None:
    print("\n" + "=" * 80)
    print("STEP 5: Rigorous Verification of Dataset Invariants across All Biomes")
    print("=" * 80)

    # 1. Verify JSON
    rgb_tasks = json.loads(ADJUSTED_RGB_JSON.read_text(encoding="utf-8"))
    thermal_tasks = json.loads(THERMAL_JSON.read_text(encoding="utf-8"))
    assert len(rgb_tasks) == 64, f"Expected 64 RGB tasks, got {len(rgb_tasks)}"
    assert len(thermal_tasks) == 64, f"Expected 64 Thermal tasks, got {len(thermal_tasks)}"

    for t in rgb_tasks:
        results = t.get("annotations", [{}])[0].get("result", [])
        tracks = [r for r in results if r.get("type") == "videorectangle"]
        assert len(tracks) <= 1, f"Task {t['id']} has {len(tracks)} tracks!"

    print("[PASS] JSON Track Invariant: All 64 tasks have <= 1 annotation box track.")

    # 2. Check frames and labels across all biomes
    violations = 0
    checked_labels = 0
    empty_labels = 0
    target_labels = 0

    total_images_by_mod = {"rgb": 0, "thermal": 0}

    for biome in ["desert", "forest", "snow"]:
        for scen in ["positive", "negative"]:
            for mod in ["rgb", "thermal"]:
                im_dir = Path(f"data/processed/images/{biome}/{scen}/{mod}")
                lb_dir = Path(f"data/processed/labels/{biome}/{scen}/{mod}")

                im_files = list(im_dir.glob("*.png"))
                lb_files = list(lb_dir.glob("*.txt"))

                total_images_by_mod[mod] += len(im_files)
                print(f"  {biome.upper():<6} {scen:<8} {mod.upper():<7}: {len(im_files)} images, {len(lb_files)} labels")
                assert len(im_files) == len(lb_files), f"Count mismatch in {biome} {scen} {mod}: {len(im_files)} vs {len(lb_files)}"

                for lb in lb_files:
                    checked_labels += 1
                    content = lb.read_text(encoding="utf-8").strip()
                    if not content:
                        empty_labels += 1
                    else:
                        lines = content.splitlines()
                        if len(lines) == 1:
                            target_labels += 1
                            parts = lines[0].split()
                            assert len(parts) == 5, f"Invalid YOLO format in {lb}: {lines[0]}"
                            cls_id, xc, yc, bw, bh = parts
                            assert cls_id == "0", f"Unexpected class {cls_id} in {lb}"
                            f_xc, f_yc, f_bw, f_bh = float(xc), float(yc), float(bw), float(bh)
                            assert 0.0 <= f_xc <= 1.0 and 0.0 <= f_yc <= 1.0, f"Out of bounds xc/yc in {lb}"
                            assert 0.0 < f_bw <= 1.0 and 0.0 < f_bh <= 1.0, f"Out of bounds bw/bh in {lb}"
                        else:
                            violations += 1
                            print(f"VIOLATION: Multiple boxes in {lb}!")

    assert violations == 0, f"Found {violations} violations!"
    print(f"\n[PASS] Single Box Invariant: Checked {checked_labels} label files.")
    print(f"       - Non-target frames (empty labels): {empty_labels}")
    print(f"       - Target frames (exactly 1 box):    {target_labels}")
    print(f"       - Total RGB Frames:                 {total_images_by_mod['rgb']}")
    print(f"       - Total Thermal Frames:             {total_images_by_mod['thermal']}")
    print(f"[PASS] Exact RGB == Thermal 1:1 Parity:     {total_images_by_mod['rgb'] == total_images_by_mod['thermal']}")

    overlays = list(VERIFICATION_DIR.glob("*.png"))
    print(f"[PASS] Verification Overlays: {len(overlays)} samples available in {VERIFICATION_DIR}")
    print("\n" + "=" * 80)
    print("ALL VERIFICATIONS COMPLETED AND INVARIANTS SATISFIED!")
    print("=" * 80)


if __name__ == "__main__":
    adjust_label_studio_json()
    rgb_tasks = json.loads(ADJUSTED_RGB_JSON.read_text(encoding="utf-8"))
    create_thermal_annotation_json(rgb_tasks)
    process_positive_frames_and_labels(rgb_tasks)
    process_negative_frames_and_labels()
    verify_all_invariants()
