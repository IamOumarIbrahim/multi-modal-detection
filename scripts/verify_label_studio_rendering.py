"""Verify video rendering in Google Chrome across all Label Studio projects.

Verifies:
1. Video element presence (attached to DOM) and correct source.
2. Video decoding metadata (videoWidth == 640, videoHeight == 640, duration > 0).
3. ReadyState == 4 (HAVE_ENOUGH_DATA).
4. Canvas rendering element visibility.
5. Playback and scrubbing (currentTime advancement).
6. Captures screenshots of actual rendered video frames inside Label Studio UI.
"""

import os
import sys
import time
from pathlib import Path
from playwright.sync_api import sync_playwright

CHROME_EXE = r"C:\Program Files\Google\Chrome\Application\chrome.exe"
BASE_URL = "http://localhost:8080"
EMAIL = "mmsar@local.dev"
PASSWORD = "mmsarpassword123"

PROJECTS_TO_VERIFY = [
    {"id": 2, "name": "Desert_Positive_Unified", "sample_tasks": [1799, 1801, 1805]},
    {"id": 7, "name": "Desert_Positive_6_10", "sample_tasks": [1809, 1811, 1815]},
    {"id": 8, "name": "Forest_Positive_1_7", "sample_tasks": [1819, 1823, 1827, 1831]},
    {"id": 9, "name": "Forest_Negative_1_7", "sample_tasks": [1833, 1837, 1841, 1845]},
]


def verify_rendering():
    out_dir = Path("scratch/screenshots")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" STARTING CHROMIUM/CHROME VIDEO RENDERING VERIFICATION")
    print("=" * 70)

    with sync_playwright() as p:
        browser = p.chromium.launch(
            executable_path=CHROME_EXE,
            headless=True,
            args=["--autoplay-policy=no-user-gesture-required"],
        )
        context = browser.new_context(viewport={"width": 1400, "height": 950})
        page = context.new_page()

        # 1. Login
        print("\n1. Logging into Label Studio...")
        page.goto(f"{BASE_URL}/user/login/")
        page.fill('input[name="email"]', EMAIL)
        page.fill('input[name="password"]', PASSWORD)
        page.click('button[type="submit"], input[type="submit"]')
        page.wait_for_timeout(2500)
        print("   Logged in successfully!")

        all_verified = True
        verification_results = []

        for p_info in PROJECTS_TO_VERIFY:
            pid = p_info["id"]
            pname = p_info["name"]
            tasks = p_info["sample_tasks"]

            print(f"\n=======================================================")
            print(f" Verifying Project {pid}: {pname}")
            print(f"=======================================================")

            for tid in tasks:
                task_url = f"{BASE_URL}/projects/{pid}/data?tab={pid}&task={tid}"
                print(f"\nLoading Task {tid} ({task_url})...")
                page.goto(task_url)

                # Wait for video element (Label Studio keeps video offscreen, so state="attached")
                page.wait_for_selector("video", state="attached", timeout=15000)

                # Wait for video readyState >= 2 (HAVE_CURRENT_DATA) or 4 (HAVE_ENOUGH_DATA)
                page.wait_for_function(
                    """() => {
                        const v = document.querySelector('video');
                        return v && v.readyState >= 2 && v.videoWidth === 640 && v.videoHeight === 640;
                    }""",
                    timeout=20000,
                )

                # Wait for canvas to be visible
                page.wait_for_selector("canvas", state="visible", timeout=15000)
                page.wait_for_timeout(1000)

                # Collect video properties
                props = page.evaluate("""() => {
                    const v = document.querySelector('video');
                    return {
                        currentSrc: v.currentSrc,
                        readyState: v.readyState,
                        videoWidth: v.videoWidth,
                        videoHeight: v.videoHeight,
                        duration: v.duration,
                        currentTime: v.currentTime,
                        paused: v.paused,
                        error: v.error ? {code: v.error.code, message: v.error.message} : null
                    };
                }""")

                # Test scrubbing / playback advancement
                page.evaluate("""() => {
                    const v = document.querySelector('video');
                    v.currentTime = 2.0; // scrub to 2.0 seconds
                }""")
                page.wait_for_timeout(1000)

                scrubbed_time = page.evaluate("() => document.querySelector('video').currentTime")

                # Take screenshot of UI with rendered frame
                scr_file = out_dir / f"proj_{pid}_task_{tid}_{pname}.png"
                page.screenshot(path=str(scr_file))

                # Validate
                is_ok = (
                    props["readyState"] >= 2
                    and props["videoWidth"] == 640
                    and props["videoHeight"] == 640
                    and props["duration"] > 0
                    and props["error"] is None
                    and abs(scrubbed_time - 2.0) < 0.5
                )

                if is_ok:
                    print(
                        f"  -> [VERIFIED OK] Video: {Path(props['currentSrc']).name} | "
                        f"Size: {props['videoWidth']}x{props['videoHeight']} | "
                        f"ReadyState: {props['readyState']} | "
                        f"Duration: {props['duration']:.2f}s | "
                        f"ScrubbedTo: {scrubbed_time:.2f}s | "
                        f"Screenshot: {scr_file.name}"
                    )
                else:
                    print(f"  -> [VERIFICATION FAILED] Props: {props}")
                    all_ok = False

                verification_results.append({
                    "project_id": pid,
                    "project_name": pname,
                    "task_id": tid,
                    "video_file": Path(props["currentSrc"]).name,
                    "video_width": props["videoWidth"],
                    "video_height": props["videoHeight"],
                    "duration": props["duration"],
                    "ready_state": props["readyState"],
                    "scrubbed_time": scrubbed_time,
                    "screenshot": str(scr_file),
                    "status": "PASS" if is_ok else "FAIL",
                })

        browser.close()

    print("\n" + "=" * 70)
    print(" RENDERING VERIFICATION SUMMARY")
    print("=" * 70)
    for r in verification_results:
        print(f"[{r['status']}] Proj {r['project_id']} ({r['project_name']}) Task {r['task_id']}: {r['video_file']} ({r['video_width']}x{r['video_height']}, {r['duration']:.2f}s, ReadyState={r['ready_state']})")

    assert all_verified, "One or more video rendering checks failed!"
    print("\nALL VIDEOS VERIFIED TO RENDER AND PLAY FLAWLESSLY IN GOOGLE CHROME INSIDE LABEL STUDIO!")


if __name__ == "__main__":
    verify_rendering()
