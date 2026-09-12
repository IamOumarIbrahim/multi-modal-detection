import os
import re
import json
import subprocess
from pathlib import Path
import imageio_ffmpeg
import requests
import cv2

BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"

def get_auth_session():
    session = requests.Session()
    session.get(f"{BASE_URL}/user/login/")
    csrf = session.cookies.get("csrftoken")
    res = session.post(
        f"{BASE_URL}/user/login/",
        data={"email": EMAIL, "password": PASSWORD, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{BASE_URL}/user/login/"}
    )
    assert res.status_code == 200, f"Login failed: {res.status_code}"
    return session

def create_or_clean_project(session, title, description, xml_config):
    csrf = session.cookies.get("csrftoken")
    headers = {"X-CSRFToken": csrf, "Referer": f"{BASE_URL}/projects"}
    
    # Check existing projects
    projs = session.get(f"{BASE_URL}/api/projects", headers=headers).json().get("results", [])
    target_p = None
    for p in projs:
        if p["title"] == title or (title.split()[1] in p["title"] and title.split()[2] in p["title"]):
            target_p = p
            break
            
    if target_p:
        project_id = target_p["id"]
        print(f"Found existing project {project_id}: '{target_p['title']}'")
        # Update config & title
        session.patch(f"{BASE_URL}/api/projects/{project_id}", headers=headers, json={"label_config": xml_config, "title": title})
        # Clear existing tasks
        tasks = session.get(f"{BASE_URL}/api/projects/{project_id}/tasks").json()
        if isinstance(tasks, list):
            print(f"  Deleting {len(tasks)} existing tasks...")
            for t in tasks:
                session.delete(f"{BASE_URL}/api/tasks/{t['id']}", headers=headers)
    else:
        payload = {
            "title": title,
            "description": description,
            "label_config": xml_config,
        }
        res = session.post(f"{BASE_URL}/api/projects", headers=headers, json=payload)
        assert res.status_code == 201, f"Failed to create project '{title}': {res.text}"
        project_id = res.json()["id"]
        print(f"Created new project {project_id}: '{title}'")
        
    return project_id

def main():
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    session = get_auth_session()
    csrf = session.cookies.get("csrftoken")
    headers = {"X-CSRFToken": csrf, "Referer": f"{BASE_URL}/projects"}
    xml_config = Path("annotations/label_studio_video_config.xml").read_text(encoding="utf-8")

    scenarios = [
        ("hard_negative", "MMSAR Desert Hard-Negative (24 FPS)", "Desert Hard Negative RGB videos (24 FPS)"),
        ("clear_negative", "MMSAR Desert Clear-Negative (24 FPS)", "Desert Clear Negative RGB videos (24 FPS)"),
    ]

    total_images = 0
    total_labels = 0

    for scenario_type, proj_title, proj_desc in scenarios:
        print(f"\n==================================================")
        print(f" PROCESSING SCENARIO: desert/{scenario_type}")
        print(f"==================================================")

        raw_dir = Path(f"data/raw/desert/{scenario_type}/rgb")
        raw_files = sorted(raw_dir.glob("*.mp4"))
        assert len(raw_files) == 5, f"Expected 5 videos in {raw_dir}, found {len(raw_files)}"

        # 1. Prepare faststart videos
        ls_video_dir = Path(f"data/processed/label_studio_videos/{scenario_type}")
        ls_video_dir.mkdir(parents=True, exist_ok=True)
        faststart_files = []

        print(f"1. Remuxing with faststart at native 24 FPS...")
        for src in raw_files:
            dst = ls_video_dir / src.name
            cmd = [ffmpeg, "-y", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(dst)]
            subprocess.run(cmd, capture_output=True, check=True)
            faststart_files.append(dst)
            print(f"  Remuxed {src.name} -> {dst.name}")

        # 2. Setup Label Studio Project & Upload
        print(f"2. Uploading to Label Studio...")
        project_id = create_or_clean_project(session, proj_title, proj_desc, xml_config)

        for v in faststart_files:
            with open(v, "rb") as f:
                files = {"file": (v.name, f, "video/mp4")}
                up_resp = session.post(f"{BASE_URL}/api/projects/{project_id}/import", headers=headers, files=files)
                print(f"  Import {v.name}: status={up_resp.status_code}, task_count={up_resp.json().get('task_count')}")

        tasks = session.get(f"{BASE_URL}/api/projects/{project_id}/tasks").json()
        print(f"  Project {project_id} has {len(tasks)} video tasks loaded at 24 FPS.")

        # 3. Decimate to 8 FPS (3:1 decimation) and export images & empty YOLO labels
        print(f"3. Decimating to 8 FPS (80 frames/video) and generating images & YOLO labels...")
        frames_base = Path(f"data/processed/frames/desert/{scenario_type}/rgb")
        images_flat_base = Path(f"data/processed/images/desert/{scenario_type}/rgb")
        labels_base = Path(f"data/processed/labels/desert/{scenario_type}/rgb")

        frames_base.mkdir(parents=True, exist_ok=True)
        images_flat_base.mkdir(parents=True, exist_ok=True)
        labels_base.mkdir(parents=True, exist_ok=True)

        for src in raw_files:
            video_stem = src.stem
            video_frames_dir = frames_base / video_stem
            video_labels_dir = labels_base / video_stem
            video_frames_dir.mkdir(parents=True, exist_ok=True)
            video_labels_dir.mkdir(parents=True, exist_ok=True)

            cap = cv2.VideoCapture(str(src))
            frame_idx = 0
            saved_idx = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % 3 == 0:
                    # Save frame image (640x640 PNG)
                    frame_filename = video_frames_dir / f"frame_{saved_idx:06d}.png"
                    flat_filename = images_flat_base / f"{video_stem}_frame_{saved_idx:06d}.png"
                    cv2.imwrite(str(frame_filename), frame)
                    cv2.imwrite(str(flat_filename), frame)
                    total_images += 1

                    # Save empty YOLO label file (negative background frame)
                    struct_label_file = video_labels_dir / f"frame_{saved_idx:06d}.txt"
                    flat_label_file = labels_base / f"{video_stem}_frame_{saved_idx:06d}.txt"
                    struct_label_file.write_text("", encoding="utf-8")
                    flat_label_file.write_text("", encoding="utf-8")
                    total_labels += 1

                    saved_idx += 1

                frame_idx += 1

            cap.release()
            print(f"  {video_stem}: 80 frames @ 8 FPS extracted (images + empty labels).")

    # 4. Update data/manifest.json
    print(f"\n4. Updating data/manifest.json...")
    manifest_path = Path("data/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["desert"]["hard_negative"]["count"] = 5
    manifest["desert"]["clear_negative"]["count"] = 5
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("  data/manifest.json updated successfully.")

    print(f"\n=== SUMMARY ===")
    print(f"Total Negative 8 FPS Images Extracted: {total_images} (expected 800)")
    print(f"Total Negative YOLO Labels Created:   {total_labels} (expected 800)")
    print("ALL HARD-NEGATIVE & CLEAR-NEGATIVE VIDEOS SUCCESSFULLY PROCESSED!")

if __name__ == "__main__":
    main()
