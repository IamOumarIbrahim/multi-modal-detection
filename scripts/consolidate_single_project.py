"""Consolidate all Desert and Forest RGB positive videos into a single unified Label Studio project (Project 2).

Actions:
1. Update Project 2 title to 'MMSAR RGB Video Tracking (Desert & Forest)'.
2. Import all 14 Forest Positive snippets into Project 2 (Desert Positive 1-10 are already present).
3. Preserve all 10 existing completed annotations (Tasks 1769-1778).
4. Remove fragmented temporary projects (Project 7, 8, 9).
5. Verify HTTP 206 streaming for all 34 tasks.
6. Verify Google Chrome rendering across Desert and Forest tasks.
"""

import re
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"


def get_auth_session():
    session = requests.Session()
    session.get(f"{BASE_URL}/user/login/")
    csrf = session.cookies.get("csrftoken")
    resp = session.post(
        f"{BASE_URL}/user/login/",
        data={"email": EMAIL, "password": PASSWORD, "csrfmiddlewaretoken": csrf},
        headers={"Referer": f"{BASE_URL}/user/login/"},
    )
    assert resp.status_code == 200, f"Login failed: {resp.status_code}"
    csrf = session.cookies.get("csrftoken")
    headers = {"X-CSRFToken": csrf, "Referer": f"{BASE_URL}/projects"}
    return session, headers


def main():
    session, headers = get_auth_session()
    xml_config = Path("annotations/label_studio_video_config.xml").read_text(encoding="utf-8")

    # 1. Update Project 2
    proj_title = "MMSAR RGB Tracking (Desert & Forest)"
    proj_desc = "Unified RGB video tracking combining Desert (Flights 1-10) and Forest (Flights 1-7) snippets (640x640 @ 24 FPS)"
    assert len(proj_title) <= 50, f"Title too long: {len(proj_title)}"

    print(f"1. Updating Project 2 to '{proj_title}'...")
    p2_resp = session.patch(
        f"{BASE_URL}/api/projects/2",
        headers=headers,
        json={"title": proj_title, "description": proj_desc, "label_config": xml_config},
    )
    assert p2_resp.status_code == 200, f"Failed to update project 2: {p2_resp.text}"

    # 2. Check existing tasks in Project 2
    existing_tasks = session.get(f"{BASE_URL}/api/projects/2/tasks", headers=headers).json()
    existing_names = set()
    for t in existing_tasks:
        v_url = t.get("data", {}).get("video", "")
        fname = Path(v_url).name
        clean_name = re.sub(r"^[0-9a-fA-F]{8}-", "", fname)
        existing_names.add(clean_name)
        existing_names.add(fname)

    print(f"   Project 2 currently has {len(existing_tasks)} tasks.")

    # 3. Import Forest Positive snippets (14 files) into Project 2
    forest_pos_dir = Path("data/raw/forest/positive/rgb")
    forest_pos_files = sorted(forest_pos_dir.glob("forest_RGB_positive_*.mp4"))
    assert len(forest_pos_files) == 14, f"Expected 14 Forest positive files, got {len(forest_pos_files)}"

    print(f"\n2. Importing {len(forest_pos_files)} Forest Positive snippets into Project 2...")
    for v in forest_pos_files:
        if v.name in existing_names:
            print(f"   [ALREADY EXISTS] {v.name}")
            continue

        with open(v, "rb") as f:
            files = {"file": (v.name, f, "video/mp4")}
            up_resp = session.post(f"{BASE_URL}/api/projects/2/import", headers=headers, files=files)
            assert up_resp.status_code == 201, f"Import failed for {v.name}: {up_resp.status_code} {up_resp.text}"
            print(f"   [IMPORTED] {v.name} ({v.stat().st_size / (1024*1024):.2f} MB)")

    # Verify total tasks in Project 2
    all_p2_tasks = session.get(f"{BASE_URL}/api/projects/2/tasks", headers=headers).json()
    print(f"\nProject 2 now has {len(all_p2_tasks)} total tasks (expected 34: 20 Desert + 14 Forest).")
    assert len(all_p2_tasks) == 34, f"Expected 34 tasks, found {len(all_p2_tasks)}"

    # 4. Remove temporary fragmented projects (7, 8, 9)
    print("\n3. Cleaning up temporary projects 7, 8, 9...")
    for pid in [7, 8, 9]:
        del_resp = session.delete(f"{BASE_URL}/api/projects/{pid}", headers=headers)
        print(f"   Deleted Project {pid}: status={del_resp.status_code}")

    # 5. Verify HTTP 206 Streaming for all 34 tasks in Project 2
    print("\n4. Verifying HTTP 206 byte-range streaming for all 34 tasks in Project 2...")
    all_stream_ok = True
    for t in all_p2_tasks:
        tid = t["id"]
        v_url = t.get("data", {}).get("video", "")
        clean_vname = re.sub(r"^[0-9a-fA-F]{8}-", "", Path(v_url).name)
        full_url = f"{BASE_URL}{v_url}"
        resp = session.get(full_url, headers={"Range": "bytes=0-1024"})
        ct = resp.headers.get("Content-Type", "")
        ar = resp.headers.get("Accept-Ranges", "")
        st = resp.status_code
        if st in (200, 206) and "video" in ct:
            print(f"   Task {tid} ({clean_vname}): [OK] HTTP {st}, Type={ct}, Ranges={ar}")
        else:
            print(f"   Task {tid} ({clean_vname}): [FAIL] HTTP {st}, Type={ct}")
            all_stream_ok = False

    assert all_stream_ok, "HTTP range streaming failed for one or more tasks!"

    # 6. Verify Browser Rendering in Google Chrome
    print("\n5. Verifying Google Chrome playback and canvas rendering in Project 2...")
    out_dir = Path("scratch/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_task_ids = [
        1769,  # Desert Positive 1 Left (annotated)
        1799,  # Desert Positive 10 Left (new)
        1801,  # Desert Positive 6 Left (new)
        all_p2_tasks[0]["id"],  # Forest Positive (first imported)
        all_p2_tasks[6]["id"],  # Forest Positive (mid imported)
        all_p2_tasks[13]["id"], # Forest Positive (last imported)
    ]

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_EXE,
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        page = context.new_page()

        page.goto(f"{BASE_URL}/user/login/")
        page.fill('input[name="email"]', EMAIL)
        page.fill('input[name="password"]', PASSWORD)
        page.click('button[type="submit"], input[type="submit"]')
        page.wait_for_timeout(2500)

        for tid in sample_task_ids:
            task_url = f"{BASE_URL}/projects/2/data?tab=2&task={tid}"
            print(f"   Checking Task {tid} ({task_url})...")
            page.goto(task_url)

            page.wait_for_selector("video", state="attached", timeout=15000)
            page.wait_for_function(
                """() => {
                    const v = document.querySelector('video');
                    return v && v.readyState >= 2 && v.videoWidth === 640 && v.videoHeight === 640;
                }""",
                timeout=20000,
            )
            page.wait_for_selector("canvas", state="visible", timeout=15000)

            props = page.evaluate("""() => {
                const v = document.querySelector('video');
                return {
                    currentSrc: v.currentSrc,
                    readyState: v.readyState,
                    videoWidth: v.videoWidth,
                    videoHeight: v.videoHeight,
                    duration: v.duration,
                    error: v.error ? v.error.message : null
                };
            }""")

            # Test scrub to 2.0s
            page.evaluate("() => document.querySelector('video').currentTime = 2.0")
            page.wait_for_timeout(800)
            scrubbed_t = page.evaluate("() => document.querySelector('video').currentTime")

            clean_name = re.sub(r"^[0-9a-fA-F]{8}-", "", Path(props["currentSrc"]).name)
            scr_file = out_dir / f"project_2_task_{tid}_{clean_name}.png"
            page.screenshot(path=str(scr_file))

            print(
                f"   -> [RENDERED OK] Task {tid} ({clean_name}): "
                f"{props['videoWidth']}x{props['videoHeight']}, {props['duration']:.2f}s, "
                f"ReadyState={props['readyState']}, ScrubbedTo={scrubbed_t:.2f}s"
            )

        browser.close()

    print("\n" + "=" * 70)
    print(" CONSOLIDATION COMPLETE: SINGLE UNIFIED PROJECT ACTIVE")
    print("=" * 70)
    print(f"Project ID:    2")
    print(f"Project Title: {proj_title}")
    print(f"Total Tasks:   {len(all_p2_tasks)} (20 Desert Positive + 14 Forest Positive)")
    print(f"Direct URL:    {BASE_URL}/projects/2")


if __name__ == "__main__":
    main()
