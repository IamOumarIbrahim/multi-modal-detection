import re
import json
from pathlib import Path
import requests
import cv2
import numpy as np
from mmsar.annotation.video_annotation_parser import interpolate_video_sequence

BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"
PROJECT_ID = 2

def main():
    print("=== STEP 1: Authenticating and Exporting from Label Studio ===")
    session = requests.Session()
    session.get(f"{BASE_URL}/user/login/")
    csrf = session.cookies.get("csrftoken")
    session.post(
        f"{BASE_URL}/user/login/",
        data={"email": EMAIL, "password": PASSWORD, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{BASE_URL}/user/login/"}
    )

    export_resp = session.get(f"{BASE_URL}/api/projects/{PROJECT_ID}/export?export_type=JSON")
    assert export_resp.status_code == 200, f"Export failed: {export_resp.status_code}"
    tasks_data = export_resp.json()
    print(f"Exported {len(tasks_data)} tasks from Project {PROJECT_ID}.")

    # Save raw export backup
    raw_dir = Path("data/annotations/raw_exports")
    raw_dir.mkdir(parents=True, exist_ok=True)
    backup_file = raw_dir / "project_2_desert_positive_export.json"
    backup_file.write_text(json.dumps(tasks_data, indent=2), encoding="utf-8")
    print(f"Saved raw export to: {backup_file}")

    # Output directories
    frames_base = Path("data/processed/frames/desert/positive/rgb")
    images_flat_base = Path("data/processed/images/desert/positive/rgb")
    labels_base = Path("data/processed/labels/desert/positive/rgb")
    debug_samples_dir = Path("data/processed/verification_samples")

    frames_base.mkdir(parents=True, exist_ok=True)
    images_flat_base.mkdir(parents=True, exist_ok=True)
    labels_base.mkdir(parents=True, exist_ok=True)
    debug_samples_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== STEP 2: Processing Videos to 8 FPS and Extracting YOLO Labels ===")
    total_images_extracted = 0
    total_labels_extracted = 0
    total_boxes_count = 0

    for task in tasks_data:
        raw_video_url = task.get("data", {}).get("video", "")
        stem_raw = Path(raw_video_url).stem
        video_stem = re.sub(r"^[0-9a-fA-F]{8}-", "", stem_raw).replace("8fps_", "")

        # Locate raw video
        raw_video_path = Path("data/raw/desert/positive/rgb") / f"{video_stem}.mp4"
        assert raw_video_path.exists(), f"Raw video missing: {raw_video_path}"

        # Get sequence annotations
        ann_list = task.get("annotations", [])
        assert len(ann_list) > 0, f"No annotations found for task {task['id']}"
        results = ann_list[0].get("result", [])
        seq = []
        if results and "value" in results[0]:
            seq = results[0]["value"].get("sequence", [])

        # Interpolate across all 240 frames of the 24 FPS video
        frame_boxes = interpolate_video_sequence(seq, total_frames=240)

        # Open video and extract at native 24 FPS (240 frames per video)
        cap = cv2.VideoCapture(str(raw_video_path))
        video_frames_dir = frames_base / video_stem
        video_labels_dir = labels_base / video_stem
        video_frames_dir.mkdir(parents=True, exist_ok=True)
        video_labels_dir.mkdir(parents=True, exist_ok=True)

        frame_idx = 0
        saved_idx = 0
        sample_saved = False

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 24 FPS frame extraction (640x640)
            frame_filename = video_frames_dir / f"frame_{saved_idx:06d}.png"
            flat_filename = images_flat_base / f"{video_stem}_frame_{saved_idx:06d}.png"
            cv2.imwrite(str(frame_filename), frame)
            cv2.imwrite(str(flat_filename), frame)
            total_images_extracted += 1

            # Matching Label Studio frame is 1-indexed (1 + frame_idx)
            ls_frame_num = frame_idx + 1
            box = frame_boxes.get(ls_frame_num)

            struct_label_file = video_labels_dir / f"frame_{saved_idx:06d}.txt"
            flat_label_file = labels_base / f"{video_stem}_frame_{saved_idx:06d}.txt"

            if box is not None:
                # Convert percentages (0..100) to normalized YOLO (0..1)
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
                total_boxes_count += 1

                # Save debug overlay for the first detected frame of each video
                if not sample_saved:
                    overlay = frame.copy()
                    ih, iw = frame.shape[:2]
                    bx1 = int((x_center - w_norm / 2.0) * iw)
                    by1 = int((y_center - h_norm / 2.0) * ih)
                    bx2 = int((x_center + w_norm / 2.0) * iw)
                    by2 = int((y_center + h_norm / 2.0) * ih)
                    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
                    cv2.putText(overlay, "Person_Detected", (bx1, max(15, by1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    sample_path = debug_samples_dir / f"overlay_{video_stem}_frame_{saved_idx:06d}.png"
                    cv2.imwrite(str(sample_path), overlay)
                    sample_saved = True
            else:
                struct_label_file.write_text("", encoding="utf-8")
                flat_label_file.write_text("", encoding="utf-8")

            total_labels_extracted += 1
            saved_idx += 1
            frame_idx += 1

        cap.release()
        print(f"  Video {video_stem}: 240 frames @ 24 FPS extracted. Saved to {video_frames_dir}")

    print("\n=== SUMMARY ===")
    print(f"Total 24 FPS images extracted: {total_images_extracted} (expected 2400)")
    print(f"Total YOLO label files:        {total_labels_extracted} (expected 2400)")
    print(f"Total positive person frames:  {total_boxes_count}")
    print(f"Verification overlays saved to: {debug_samples_dir}")
    print("COMPLETE!")

if __name__ == "__main__":
    main()

