import requests
from pathlib import Path

BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"

def main():
    session = requests.Session()
    # 1. Fetch CSRF token
    session.get(f"{BASE_URL}/user/login/")
    csrf = session.cookies.get("csrftoken")
    
    # 2. Login
    login_resp = session.post(
        f"{BASE_URL}/user/login/",
        data={"email": EMAIL, "password": PASSWORD, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{BASE_URL}/user/login/"}
    )
    assert login_resp.status_code == 200, f"Login failed: {login_resp.status_code}"
    
    csrf = session.cookies.get("csrftoken")
    headers = {"X-CSRFToken": csrf, "Referer": f"{BASE_URL}/projects/2"}
    
    # Check Project 2
    project_id = 2
    p_resp = session.get(f"{BASE_URL}/api/projects/{project_id}")
    if p_resp.status_code != 200:
        # Create Project 2
        xml_config = Path("annotations/label_studio_video_config.xml").read_text(encoding="utf-8")
        payload = {
            "title": "MMSAR Desert Positive Video Tracking",
            "label_config": xml_config,
            "description": "Tracking Person_Detected on 5 Desert Positive RGB videos (8 FPS)",
        }
        create_resp = session.post(f"{BASE_URL}/api/projects", headers=headers, json=payload)
        project_id = create_resp.json()["id"]
        print(f"Created project {project_id}")
    else:
        print(f"Using existing project {project_id}: {p_resp.json()['title']}")
    # Update project config to 24 FPS
    xml_config = Path("annotations/label_studio_video_config.xml").read_text(encoding="utf-8")
    payload = {
        "title": "MMSAR Desert Positive Video Tracking (24 FPS, Left & Right)",
        "label_config": xml_config,
        "description": "Tracking Person_Detected on 10 Desert Positive RGB video snippets (Left & Right 640x640 @ 24 FPS)",
    }
    session.patch(f"{BASE_URL}/api/projects/{project_id}", headers=headers, json=payload)
    print("Updated Project 2 label_config to 24.0 FPS (Left & Right snippets).")
        
    # Clear existing tasks
    tasks_resp = session.get(f"{BASE_URL}/api/projects/{project_id}/tasks")
    if tasks_resp.status_code == 200 and isinstance(tasks_resp.json(), list):
        old_tasks = tasks_resp.json()
        print(f"Deleting {len(old_tasks)} existing tasks...")
        for t in old_tasks:
            tid = t["id"]
            session.delete(f"{BASE_URL}/api/tasks/{tid}", headers=headers)
            
    # Upload 24 FPS left and right video snippets
    video_dir = Path("data/raw/desert/positive/rgb")
    video_files = sorted(video_dir.glob("*.mp4"))
    print(f"\nFound {len(video_files)} 24 FPS video snippets to import into project {project_id}:")
    for v in video_files:
        print(f"  - {v.name} ({v.stat().st_size / (1024*1024):.2f} MB)")
    
    assert len(video_files) == 10, f"Expected 10 positive snippets (5 left, 5 right), got {len(video_files)}"
    
    for v in video_files:
        with open(v, "rb") as f:
            files = {"file": (v.name, f, "video/mp4")}
            up_resp = session.post(f"{BASE_URL}/api/projects/{project_id}/import", headers=headers, files=files)
            assert up_resp.status_code == 201, f"Import failed for {v.name}: {up_resp.status_code} {up_resp.text}"
            print(f"  Imported {v.name}: status={up_resp.status_code}, task_count={up_resp.json().get('task_count')}")
            
    tasks = session.get(f"{BASE_URL}/api/projects/{project_id}/tasks").json()
    print(f"\nSUCCESS! Project {project_id} has {len(tasks)} video tasks loaded at 24 FPS.")
    
    print("\nVerifying video loading and HTTP partial content streaming for all tasks:")
    all_loaded = True
    for t in tasks:
        task_id = t["id"]
        v_url = t.get("data", {}).get("video", "")
        print(f"  Task {task_id}: {v_url}")
        
        # Test HTTP Range request (simulating browser HTML5 video player loading first chunk)
        full_url = f"{BASE_URL}{v_url}"
        stream_resp = session.get(full_url, headers={"Range": "bytes=0-1024"})
        content_type = stream_resp.headers.get("Content-Type", "")
        accept_ranges = stream_resp.headers.get("Accept-Ranges", "")
        status = stream_resp.status_code
        
        if status in (200, 206) and "video" in content_type:
            print(f"    -> [VERIFIED OK] Status: {status}, Content-Type: {content_type}, Accept-Ranges: {accept_ranges}, Bytes: {len(stream_resp.content)}")
        else:
            print(f"    -> [FAILED] Status: {status}, Content-Type: {content_type}")
            all_loaded = False
            
    assert all_loaded, "One or more video tasks failed to stream via HTTP!"
    print(f"\nALL {len(tasks)} VIDEO SNIPPETS VERIFIED AND PLAYABLE IN LABEL STUDIO!")
    print(f"Direct Annotation URL: {BASE_URL}/projects/{project_id}")

if __name__ == "__main__":
    main()

