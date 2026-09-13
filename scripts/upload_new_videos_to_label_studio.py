"""Upload new Desert and Forest RGB videos to Label Studio and verify streaming/rendering.

Uploads:
1. Desert Positive Flights 6-10 (10 snippets) -> appends to Project 2 and dedicated standalone project.
2. Forest Positive Flights 1-7 (14 snippets) -> creates MMSAR Forest Positive project.
3. Forest Negative Flights 1-7 (14 snippets) -> creates MMSAR Forest Negative project.
"""

import os
import re
import sys
import time
from pathlib import Path
import requests

BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"


def get_authenticated_session() -> tuple[requests.Session, dict[str, str]]:
    session = requests.Session()
    session.get(f"{BASE_URL}/user/login/")
    csrf = session.cookies.get("csrftoken")
    login_resp = session.post(
        f"{BASE_URL}/user/login/",
        data={"email": EMAIL, "password": PASSWORD, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{BASE_URL}/user/login/"},
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.status_code}"
    csrf = session.cookies.get("csrftoken")
    headers = {"X-CSRFToken": csrf, "Referer": f"{BASE_URL}/projects"}
    return session, headers


def get_or_create_project(
    session: requests.Session,
    headers: dict[str, str],
    title: str,
    description: str,
    xml_config: str,
) -> int:
    assert len(title) <= 50, f"Title exceeds 50 chars ({len(title)}): {title}"
    projs = session.get(f"{BASE_URL}/api/projects", headers=headers).json().get("results", [])
    for p in projs:
        if p.get("title") == title:
            project_id = p.get("id")
            print(f"Found existing project {project_id}: '{title}'")
            session.patch(
                f"{BASE_URL}/api/projects/{project_id}",
                headers=headers,
                json={"title": title, "description": description, "label_config": xml_config},
            )
            return project_id

    payload = {"title": title, "description": description, "label_config": xml_config}
    resp = session.post(f"{BASE_URL}/api/projects", headers=headers, json=payload)
    assert resp.status_code == 201, f"Project creation failed: {resp.text}"
    project_id = resp.json()["id"]
    print(f"Created new project {project_id}: '{title}'")
    return project_id


def upload_videos_to_project(
    session: requests.Session,
    headers: dict[str, str],
    project_id: int,
    video_files: list[Path],
) -> list[int]:
    tasks = session.get(f"{BASE_URL}/api/projects/{project_id}/tasks", headers=headers).json()
    existing_filenames = set()
    if isinstance(tasks, list):
        for t in tasks:
            v_url = t.get("data", {}).get("video", "")
            fname = Path(v_url).name
            clean_name = re.sub(r"^[0-9a-fA-F]{8}-", "", fname)
            existing_filenames.add(clean_name)
            existing_filenames.add(fname)

    print(f"\nProject {project_id}: Target videos: {len(video_files)} ({len(existing_filenames)} already in project)...")
    
    for v in video_files:
        if v.name in existing_filenames:
            print(f"  [ALREADY EXISTS] {v.name} in Project {project_id}")
            continue

        with open(v, "rb") as f:
            files = {"file": (v.name, f, "video/mp4")}
            up_resp = session.post(f"{BASE_URL}/api/projects/{project_id}/import", headers=headers, files=files)
            assert up_resp.status_code == 201, f"Import failed for {v.name}: {up_resp.status_code} {up_resp.text}"
            print(f"  [IMPORTED] {v.name} ({v.stat().st_size / (1024*1024):.2f} MB): task_count={up_resp.json().get('task_count')}")

    updated_tasks = session.get(f"{BASE_URL}/api/projects/{project_id}/tasks", headers=headers).json()
    return updated_tasks


def verify_project_streaming(session: requests.Session, project_id: int, project_name: str) -> bool:
    tasks = session.get(f"{BASE_URL}/api/projects/{project_id}/tasks").json()
    print(f"\nVerifying HTTP range streaming for Project {project_id} ('{project_name}'), total tasks: {len(tasks)}")
    all_ok = True
    for t in tasks:
        tid = t["id"]
        v_url = t.get("data", {}).get("video", "")
        full_url = f"{BASE_URL}{v_url}"
        resp = session.get(full_url, headers={"Range": "bytes=0-1024"})
        ct = resp.headers.get("Content-Type", "")
        ar = resp.headers.get("Accept-Ranges", "")
        st = resp.status_code
        if st in (200, 206) and "video" in ct:
            print(f"  Task {tid} ({Path(v_url).name}): [OK] HTTP {st}, Type={ct}, Ranges={ar}, Bytes={len(resp.content)}")
        else:
            print(f"  Task {tid} ({Path(v_url).name}): [FAIL] HTTP {st}, Type={ct}")
            all_ok = False
    return all_ok


def main():
    session, headers = get_authenticated_session()
    xml_config = Path("annotations/label_studio_video_config.xml").read_text(encoding="utf-8")

    # 1. New Desert Positive Videos (Flights 6 to 10)
    desert_pos_dir = Path("data/raw/desert/positive/rgb")
    desert_pos_all_new = [
        f for f in sorted(desert_pos_dir.glob("desert_RGB_positive_*.mp4"))
        if any(f"positive_{i}_" in f.name for i in range(6, 11))
    ]
    print(f"Found {len(desert_pos_all_new)} new Desert Positive snippets to annotate (Flights 6-10):")
    for f in desert_pos_all_new:
        print(f"  - {f.name}")
    assert len(desert_pos_all_new) == 10, f"Expected 10 new Desert snippets, got {len(desert_pos_all_new)}"

    # A) Update Project 2 (unified desert positive, max 50 chars)
    p2_title = "MMSAR Desert Positive (Flights 1-10)"
    session.patch(
        f"{BASE_URL}/api/projects/2",
        headers=headers,
        json={
            "title": p2_title,
            "description": "Tracking Person_Detected on 20 Desert Positive RGB video snippets (Left & Right 640x640 @ 24 FPS, Flights 1 to 10)",
            "label_config": xml_config,
        }
    )
    p2_tasks = upload_videos_to_project(session, headers, project_id=2, video_files=desert_pos_all_new)
    print(f"Project 2 now has {len(p2_tasks)} total tasks (10 existing annotated + 10 new to annotate).")

    # B) Also create a dedicated project for Flights 6-10 (max 50 chars)
    p_desert_new_title = "MMSAR Desert Positive 6-10 (24 FPS)"
    p_desert_new_id = get_or_create_project(
        session,
        headers,
        title=p_desert_new_title,
        description="Tracking Person_Detected on 10 new Desert Positive RGB video snippets (Flights 6-10, Left & Right 640x640 @ 24 FPS)",
        xml_config=xml_config,
    )
    p_desert_new_tasks = upload_videos_to_project(session, headers, project_id=p_desert_new_id, video_files=desert_pos_all_new)
    assert len(p_desert_new_tasks) == 10, f"Expected 10 tasks in Desert 6-10 project, got {len(p_desert_new_tasks)}"

    # 2. Forest Positive Videos (Flights 1 to 7)
    forest_pos_dir = Path("data/raw/forest/positive/rgb")
    forest_pos_files = sorted(forest_pos_dir.glob("forest_RGB_positive_*.mp4"))
    print(f"\nFound {len(forest_pos_files)} Forest Positive snippets to annotate (Flights 1-7):")
    for f in forest_pos_files:
        print(f"  - {f.name}")
    assert len(forest_pos_files) == 14, f"Expected 14 Forest Positive snippets, got {len(forest_pos_files)}"

    p_forest_pos_title = "MMSAR Forest Positive (24 FPS)"
    p_forest_pos_id = get_or_create_project(
        session,
        headers,
        title=p_forest_pos_title,
        description="Tracking Person_Detected on 14 Forest Positive RGB video snippets (Flights 1-7, Left & Right 640x640 @ 24 FPS)",
        xml_config=xml_config,
    )
    p_forest_pos_tasks = upload_videos_to_project(session, headers, project_id=p_forest_pos_id, video_files=forest_pos_files)
    assert len(p_forest_pos_tasks) == 14, f"Expected 14 tasks in Forest Positive project, got {len(p_forest_pos_tasks)}"

    # 3. Forest Negative Videos (Flights 1 to 7)
    forest_neg_dir = Path("data/raw/forest/negative/rgb")
    forest_neg_files = sorted(forest_neg_dir.glob("forest_RGB_negative_*.mp4"))
    print(f"\nFound {len(forest_neg_files)} Forest Negative snippets for review (Flights 1-7):")
    for f in forest_neg_files:
        print(f"  - {f.name}")
    assert len(forest_neg_files) == 14, f"Expected 14 Forest Negative snippets, got {len(forest_neg_files)}"

    p_forest_neg_title = "MMSAR Forest Negative (24 FPS)"
    p_forest_neg_id = get_or_create_project(
        session,
        headers,
        title=p_forest_neg_title,
        description="Reviewing 14 Forest Negative RGB video snippets (Flights 1-7, Left & Right 640x640 @ 24 FPS)",
        xml_config=xml_config,
    )
    p_forest_neg_tasks = upload_videos_to_project(session, headers, project_id=p_forest_neg_id, video_files=forest_neg_files)
    assert len(p_forest_neg_tasks) == 14, f"Expected 14 tasks in Forest Negative project, got {len(p_forest_neg_tasks)}"

    # 4. Verification of HTTP Range Streaming
    v1 = verify_project_streaming(session, 2, p2_title)
    v2 = verify_project_streaming(session, p_desert_new_id, p_desert_new_title)
    v3 = verify_project_streaming(session, p_forest_pos_id, p_forest_pos_title)
    v4 = verify_project_streaming(session, p_forest_neg_id, p_forest_neg_title)

    assert v1 and v2 and v3 and v4, "HTTP range streaming verification failed for one or more projects!"

    print("\n=================================================================")
    print(" ALL NEW VIDEOS SUCCESSFULLY ADDED AND HTTP STREAMING VERIFIED! ")
    print("=================================================================")
    print(f"- Desert Positive Unified Project 2:  {len(p2_tasks)} tasks ({BASE_URL}/projects/2)")
    print(f"- Desert Positive Flights 6-10 Proj {p_desert_new_id}: {len(p_desert_new_tasks)} tasks ({BASE_URL}/projects/{p_desert_new_id})")
    print(f"- Forest Positive Flights 1-7  Proj {p_forest_pos_id}: {len(p_forest_pos_tasks)} tasks ({BASE_URL}/projects/{p_forest_pos_id})")
    print(f"- Forest Negative Flights 1-7  Proj {p_forest_neg_id}: {len(p_forest_neg_tasks)} tasks ({BASE_URL}/projects/{p_forest_neg_id})")


if __name__ == "__main__":
    main()
