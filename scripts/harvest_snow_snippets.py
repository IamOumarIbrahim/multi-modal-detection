"""Harvest dual 640x640 snippets (left and right) from Snow RGB videos in Downloads.

Processes:
- Positives: Snow_RGB_positive_1 to 8 (1280x720 -> 2x 640x640, 240 frames @ 24 FPS)

Outputs to:
- data/raw/snow/positive/rgb/
"""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
import cv2
import imageio_ffmpeg


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps if fps > 0 else 0
    return w, h, fps, frames, duration


def harvest_snow_snippets(
    source_dir: Path = Path(r"D:\Downloads"),
    dest_base: Path = Path("data/raw/snow/positive/rgb"),
):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    start_time = time.time()
    dest_base.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(" STARTING SNOW DUAL-CROP HARVESTING PIPELINE")
    print("=" * 70)
    print(f"Source: {source_dir}")
    print(f"Destination: {dest_base}\n")

    input_files = [f"Snow_RGB_positive_{i}.mp4" for i in range(1, 9)]
    harvested = []

    for fname in input_files:
        video_path = source_dir / fname
        if not video_path.exists():
            print(f"[ERROR] Source video not found: {video_path}")
            continue

        w, h, fps, frames, dur = get_video_info(video_path)
        print(f"Source Video: {fname:<25} ({w}x{h}, {fps:.1f} fps, {frames} frames, {dur:.2f}s)")
        assert w == 1280 and h == 720, f"Unexpected dimensions {w}x{h} for {fname}"

        # Standard 1280x720 -> 2x 640x640 (centered vertically: y=40)
        left_filter = "crop=640:640:0:40"
        right_filter = "crop=640:640:640:40"

        # Canonical naming: snow_RGB_positive_{i}_{side}.mp4
        i_num = fname.split("_")[-1].replace(".mp4", "")
        crops = [
            ("left", left_filter, dest_base / f"snow_RGB_positive_{i_num}_left.mp4"),
            ("right", right_filter, dest_base / f"snow_RGB_positive_{i_num}_right.mp4"),
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
                print(f"  [FAILED] {out_path.name}: {res.stderr[:200]}")
                continue

            cw, ch, cfps, cframes, cdur = get_video_info(out_path)
            sz_mb = out_path.stat().st_size / (1024 * 1024)
            print(f"  -> Generated: {out_path.name:<32} ({cw}x{ch}, {cframes} frames, {sz_mb:.2f} MB)")
            assert cw == 640 and ch == 640, f"Crop dimension mismatch: {cw}x{ch}"
            assert cframes == frames, f"Frame count mismatch: {cframes} vs {frames}"
            harvested.append(out_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(f"HARVEST COMPLETE: {len(harvested)} snippets generated in {elapsed:.1f}s")
    print("=" * 70)
    return harvested


if __name__ == "__main__":
    harvest_snow_snippets()
