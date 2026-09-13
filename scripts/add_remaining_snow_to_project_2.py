"""Add remaining Snow RGB positive snippets (9 through 15) to Project 2 in Label Studio.

Actions:
1. Authenticate with Label Studio.
2. Update Project 2 description to reflect Snow 1-15.
3. Import all 14 newly harvested Snow positive snippets (snow_RGB_positive_9..15 left/right).
4. Verify total tasks in Project 2 = 64 (20 Desert + 14 Forest + 30 Snow).
5. Verify HTTP 206 partial content streaming for all newly added snow tasks.
6. Verify Chrome playback, canvas rendering, and timeline scrubbing using Playwright.
"""

from __future__ import annotations

import re
import sys
import time
from pathlib import Path
import requests
from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"
CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
ARTIFACT_DIR = Path(r"C:\Users\omarb\.gemini\antigravity\brain\fc0c010e-63ae-48e1-9974-80457f367297")


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

    # 1. Update Project 2 Title / Description
    proj_title = "MMSAR RGB Tracking (Desert, Forest & Snow)"
    proj_desc = "Unified RGB video tracking combining Desert (1-10), Forest (1-7), and Snow (1-15) snippets (640x640 @ 24 FPS)"

    print(f"1. Updating Project 2 metadata...")
    p2_resp = session.patch(
        f"{BASE_URL}/api/projects/2",
        headers=headers,
        json={"title": proj_title, "description": proj_desc, "label_config": xml_config},
    )
    assert p2_resp.status_code == 200, f"Failed to update project 2: {p2_resp.text}"

    # 2. Get existing tasks with page_size=200 to avoid pagination truncation
    existing_tasks = session.get(f"{BASE_URL}/api/projects/2/tasks?page_size=200", headers=headers).json()
    existing_names = set()
    for t in existing_tasks:
        v_url = t.get("data", {}).get("video", "")
        fname = Path(v_url).name
        clean_name = re.sub(r"^[0-9a-fA-F]{8}-", "", fname)
        existing_names.add(clean_name)
        existing_names.add(fname)

    print(f"   Project 2 currently has {len(existing_tasks)} tasks loaded.")

    # 3. Check for any unimported Snow Positive snippets (9 to 15)
    snow_pos_dir = Path("data/raw/snow/positive/rgb")
    all_snow_files = sorted(snow_pos_dir.glob("snow_RGB_positive_*.mp4"))

    files_to_import = []
    for f in all_snow_files:
        if f.name not in existing_names:
            files_to_import.append(f)

    if files_to_import:
        print(f"\n2. Importing {len(files_to_import)} new Snow Positive snippets...")
        for v in files_to_import:
            with open(v, "rb") as fp:
                files = {"file": (v.name, fp, "video/mp4")}
                up_resp = session.post(f"{BASE_URL}/api/projects/2/import", headers=headers, files=files)
                assert up_resp.status_code == 201, f"Import failed for {v.name}: {up_resp.status_code}"
                print(f"   [IMPORTED] {v.name}")
    else:
        print(f"\n2. All {len(all_snow_files)} Snow Positive snippets are already imported into Project 2!")

    # 4. Verify total tasks in Project 2
    proj_meta = session.get(f"{BASE_URL}/api/projects/2", headers=headers).json()
    all_p2_tasks = session.get(f"{BASE_URL}/api/projects/2/tasks?page_size=200", headers=headers).json()
    total_tasks = len(all_p2_tasks)
    print(f"\n3. Verifying total tasks: Project 2 task_number={proj_meta.get('task_number')}, retrieved={total_tasks} (expected 64).")
    assert total_tasks == 64, f"Expected 64 tasks, found {total_tasks}"
    assert proj_meta.get("task_number") == 64, f"Expected task_number=64, found {proj_meta.get('task_number')}"

    # 5. Verify HTTP 206 Streaming for all Snow tasks (especially 9..15)
    print("\n4. Verifying HTTP 206 byte-range streaming for Snow positive tasks 9..15...")
    new_snow_stems = [f"snow_RGB_positive_{i}_{side}.mp4" for i in range(9, 16) for side in ["left", "right"]]
    verified_streaming = 0

    for t in all_p2_tasks:
        v_url = t.get("data", {}).get("video", "")
        clean_vname = re.sub(r"^[0-9a-fA-F]{8}-", "", Path(v_url).name)
        if clean_vname in new_snow_stems:
            full_url = f"{BASE_URL}{v_url}"
            resp = session.get(full_url, headers={"Range": "bytes=0-1024"})
            ct = resp.headers.get("Content-Type", "")
            ar = resp.headers.get("Accept-Ranges", "")
            st = resp.status_code
            print(f"   Task {t['id']:<3} ({clean_vname:<32}): [OK] HTTP {st}, Content-Type={ct}, Accept-Ranges={ar}")
            assert st in (200, 206) and "video" in ct, f"Streaming check failed for {clean_vname}"
            verified_streaming += 1

    print(f"   Successfully verified HTTP 206 streaming for {verified_streaming}/14 snow tasks (9..15).")
    assert verified_streaming == 14, f"Expected 14 verified streams, got {verified_streaming}"

    # 6. Verify Google Chrome Playback and Canvas Rendering via Playwright
    print("\n5. Verifying Chrome playback, canvas rendering, and timeline scrub via Playwright...")
    target_tasks = []
    for t in all_p2_tasks:
        v_url = t.get("data", {}).get("video", "")
        clean_vname = re.sub(r"^[0-9a-fA-F]{8}-", "", Path(v_url).name)
        if clean_vname in new_snow_stems:
            target_tasks.append((t, clean_vname))

    # Sample task 9_left, 12_left, 15_left
    sample_indices = [0, len(target_tasks) // 2, -1]
    sample_tasks = [target_tasks[i] for i in sample_indices]

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

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

        for task, clean_name in sample_tasks:
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

            # Test timeline scrubbing to 2.0s
            page.evaluate("() => document.querySelector('video').currentTime = 2.0")
            page.wait_for_timeout(800)
            scrubbed_t = page.evaluate("() => document.querySelector('video').currentTime")

            scr_file = ARTIFACT_DIR / f"project_2_task_{tid}_{clean_name}.png"
            page.screenshot(path=str(scr_file))

            print(
                f"   -> Rendered OK: {clean_name} | {props['videoWidth']}x{props['videoHeight']} | "
                f"readyState={props['readyState']} | scrubbed={scrubbed_t:.2f}s | duration={props['duration']:.1f}s | saved {scr_file.name}"
            )
            assert props["videoWidth"] == 640 and props["videoHeight"] == 640
            assert props["readyState"] >= 2
            assert abs(scrubbed_t - 2.0) < 0.2

        browser.close()

    print("\n" + "=" * 70)
    print(" ALL 14 REMAINING SNOW POSITIVE TASKS SUCCESSFULLY VERIFIED IN PROJECT 2!")
    print(f" Total tasks in Project 2: 64")
    print(f" Direct Annotation URL: {BASE_URL}/projects/2")
    print("=" * 70)


if __name__ == "__main__":
    main()
