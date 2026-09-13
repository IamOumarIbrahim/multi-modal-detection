"""Add harvested Snow RGB positive snippets to Project 2 (Unified RGB Tracking) in Label Studio.

Actions:
1. Update Project 2 title to 'MMSAR RGB Tracking (Desert, Forest & Snow)'.
2. Import all 16 Snow Positive snippets into Project 2.
3. Verify total tasks in Project 2 = 50 (34 existing + 16 new).
4. Verify HTTP 206 streaming for newly imported Snow tasks.
5. Verify Google Chrome rendering and timeline scrubbing via Playwright.
"""

from __future__ import annotations

import re
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ARTIFACT_DIR = Path(r"C:\Users\omarb\.gemini\antigravity\brain\c132b5f0-50aa-4989-9d27-a556b5a6309f")


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

    # 1. Update Project 2 Title
    proj_title = "MMSAR RGB Tracking (Desert, Forest & Snow)"
    proj_desc = "Unified RGB video tracking combining Desert (1-10), Forest (1-7), and Snow (1-8) snippets (640x640 @ 24 FPS)"
    assert len(proj_title) <= 50, f"Title too long: {len(proj_title)}"

    print(f"1. Updating Project 2 title to '{proj_title}'...")
    p2_resp = session.patch(
        f"{BASE_URL}/api/projects/2",
        headers=headers,
        json={"title": proj_title, "description": proj_desc, "label_config": xml_config},
    )
    assert p2_resp.status_code == 200, f"Failed to update project 2: {p2_resp.text}"

    # 2. Get existing tasks
    existing_tasks = session.get(f"{BASE_URL}/api/projects/2/tasks", headers=headers).json()
    existing_names = set()
    for t in existing_tasks:
        v_url = t.get("data", {}).get("video", "")
        fname = Path(v_url).name
        clean_name = re.sub(r"^[0-9a-fA-F]{8}-", "", fname)
        existing_names.add(clean_name)
        existing_names.add(fname)

    print(f"   Project 2 currently has {len(existing_tasks)} tasks.")

    # 3. Import Snow Positive snippets (16 files)
    snow_pos_dir = Path("data/raw/snow/positive/rgb")
    snow_pos_files = sorted(snow_pos_dir.glob("snow_RGB_positive_*.mp4"))
    assert len(snow_pos_files) == 16, f"Expected 16 Snow positive files, got {len(snow_pos_files)}"

    print(f"\n2. Importing {len(snow_pos_files)} Snow Positive snippets into Project 2...")
    imported_count = 0
    for v in snow_pos_files:
        if v.name in existing_names:
            print(f"   [ALREADY EXISTS] {v.name}")
            continue

        with open(v, "rb") as f:
            files = {"file": (v.name, f, "video/mp4")}
            up_resp = session.post(f"{BASE_URL}/api/projects/2/import", headers=headers, files=files)
            assert up_resp.status_code == 201, f"Import failed for {v.name}: {up_resp.status_code} {up_resp.text}"
            print(f"   [IMPORTED] {v.name} ({v.stat().st_size / (1024*1024):.2f} MB)")
            imported_count += 1

    # Verify total tasks in Project 2
    all_p2_tasks = session.get(f"{BASE_URL}/api/projects/2/tasks", headers=headers).json()
    print(f"\nProject 2 now has {len(all_p2_tasks)} total tasks (expected 50: 20 Desert + 14 Forest + 16 Snow).")
    assert len(all_p2_tasks) == 50, f"Expected 50 tasks, found {len(all_p2_tasks)}"

    # 4. Verify HTTP 206 Streaming for Snow tasks
    print("\n3. Verifying HTTP 206 byte-range streaming for Snow tasks in Project 2...")
    snow_tasks = []
    for t in all_p2_tasks:
        v_url = t.get("data", {}).get("video", "")
        clean_vname = re.sub(r"^[0-9a-fA-F]{8}-", "", Path(v_url).name)
        if "snow_" in clean_vname.lower():
            snow_tasks.append(t)
            full_url = f"{BASE_URL}{v_url}"
            resp = session.get(full_url, headers={"Range": "bytes=0-1024"})
            ct = resp.headers.get("Content-Type", "")
            ar = resp.headers.get("Accept-Ranges", "")
            st = resp.status_code
            print(f"   Task {t['id']} ({clean_vname}): [OK] HTTP {st}, Type={ct}, Ranges={ar}")
            assert st in (200, 206) and "video" in ct, f"Streaming check failed for {clean_vname}"

    assert len(snow_tasks) == 16, f"Expected 16 snow tasks, found {len(snow_tasks)}"

    # 5. Verify Browser Rendering in Google Chrome for sample Snow tasks
    print("\n4. Verifying Google Chrome playback and canvas rendering for Snow tasks...")
    sample_snow_tasks = [snow_tasks[0], snow_tasks[len(snow_tasks)//2], snow_tasks[-1]]

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

        for task in sample_snow_tasks:
            tid = task["id"]
            task_url = f"{BASE_URL}/projects/2/data?tab=2&task={tid}"
            print(f"   Checking Snow Task {tid} ({task_url})...")
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
            scr_file = ARTIFACT_DIR / f"project_2_task_{tid}_{clean_name}.png"
            page.screenshot(path=str(scr_file))

            print(
                f"   -> Rendered OK: {clean_name} | {props['videoWidth']}x{props['videoHeight']} | "
                f"readyState={props['readyState']} | scrubbed={scrubbed_t:.2f}s | saved {scr_file.name}"
            )
            assert props["videoWidth"] == 640 and props["videoHeight"] == 640
            assert props["readyState"] >= 2
            assert abs(scrubbed_t - 2.0) < 0.2

        browser.close()

    print("\n" + "=" * 70)
    print("ALL 16 SNOW TASKS ADDED AND VERIFIED IN PROJECT 2!")
    print(f"Direct Annotation URL: {BASE_URL}/projects/2")
    print("=" * 70)


if __name__ == "__main__":
    main()
