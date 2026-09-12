"""Harvest dual 640x640 snippets (left and right) from 1280x720 desert RGB videos.

This script executes the dataset 2x size procedure:
- Input: 15 raw videos in videos/desert/{scenario}/rgb/ (1280x720, 24 FPS, 10s)
- Vertical crop: centered vertically, offset y = (720 - 640) / 2 = 40
- Left snippet: x = 0, y = 40, 640x640
- Right snippet: x = 640, y = 40, 640x640
- Output: 30 snippets in data/raw/desert/{scenario}/rgb/ (640x640, 24 FPS, 10s, +faststart)
"""

import subprocess
import time
from pathlib import Path
import cv2
import imageio_ffmpeg

SCENARIOS = ["clear_negative", "hard_negative", "positive"]

def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps if fps > 0 else 0
    return w, h, fps, frames, duration

def harvest_snippets(
    source_base: Path = Path("videos/desert"),
    dest_base: Path = Path("data/raw/desert"),
):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    total_processed = 0
    total_harvested = 0
    start_time = time.time()

    print("================================================================")
    print(" STARTING DATASET 2X DUAL-CROP HARVESTING PROCEDURE")
    print("================================================================")

    for scenario in SCENARIOS:
        src_dir = source_base / scenario / "rgb"
        out_dir = dest_base / scenario / "rgb"
        out_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(src_dir.glob("*.mp4"))
        print(f"\nScenario: {scenario} -> Found {len(video_files)} videos in {src_dir}")

        for video_path in video_files:
            stem = video_path.stem
            w, h, fps, frames, dur = get_video_info(video_path)
            print(f"\nProcessing {video_path.name} ({w}x{h}, {fps:.1f} fps, {frames} frames, {dur:.2f}s):")

            assert w == 1280 and h == 720, f"Expected 1280x720, got {w}x{h} for {video_path}"

            crops = [
                ("left", "crop=640:640:0:40", out_dir / f"{stem}_left.mp4"),
                ("right", "crop=640:640:640:40", out_dir / f"{stem}_right.mp4"),
            ]

            for side, crop_filter, out_path in crops:
                cmd = [
                    ffmpeg_exe,
                    "-y",
                    "-i",
                    str(video_path),
                    "-vf",
                    crop_filter,
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-preset",
                    "fast",
                    "-pix_fmt",
                    "yuv420p",
                    "-r",
                    "24",
                    "-movflags",
                    "+faststart",
                    str(out_path),
                ]

                res = subprocess.run(cmd, capture_output=True, text=True)
                if res.returncode != 0:
                    raise RuntimeError(f"FFmpeg failed for {out_path}:\n{res.stderr}")

                ow, oh, ofps, oframes, odur = get_video_info(out_path)
                size_mb = out_path.stat().st_size / (1024 * 1024)
                print(f"  [{side.upper()}] -> {out_path.name}: {ow}x{oh}, {ofps:.1f} fps, {oframes} frames, {odur:.2f}s ({size_mb:.2f} MB)")
                
                assert ow == 640 and oh == 640, f"Invalid dimensions: {ow}x{oh}"
                assert oframes == 240, f"Invalid frame count: {oframes}"
                assert abs(ofps - 24.0) < 0.1, f"Invalid FPS: {ofps}"

                total_harvested += 1

            total_processed += 1

    elapsed = time.time() - start_time
    print("\n================================================================")
    print(" HARVESTING SUMMARY")
    print("================================================================")
    print(f"Raw Videos Processed:    {total_processed} (expected 15)")
    print(f"Snippets Harvested:      {total_harvested} (expected 30)")
    print(f"Elapsed Time:            {elapsed:.2f} seconds")
    print("SUCCESS: All 30 left and right 640x640 24fps snippets created and verified!")

if __name__ == "__main__":
    harvest_snippets()
