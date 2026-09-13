"""Convert all raw RGB video snippets in data/raw to Thermal using Melobytes Medium-Low LUT.

Processes:
- data/raw/desert/negative/rgb/  -> data/raw/desert/negative/thermal/
- data/raw/desert/positive/rgb/  -> data/raw/desert/positive/thermal/
- data/raw/forest/negative/rgb/  -> data/raw/forest/negative/thermal/
- data/raw/forest/positive/rgb/  -> data/raw/forest/positive/thermal/

Each RGB video frame is mapped through the 256-level Melobytes thermal LUT
and encoded to H.264 (CRF 18, 24 FPS, +faststart).
"""

import os
import re
import sys
import time
import subprocess
from pathlib import Path
import cv2
import numpy as np
import imageio_ffmpeg


def get_video_info(video_path: Path):
    cap = cv2.VideoCapture(str(video_path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps if fps > 0 else 0
    return w, h, fps, frames, duration


def convert_video(in_path: Path, out_path: Path, lut: np.ndarray, ffmpeg_exe: str):
    w, h, fps, total_frames, dur = get_video_info(in_path)

    cmd = [
        ffmpeg_exe,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{w}x{h}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(fps),
        "-i",
        "-",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-movflags",
        "+faststart",
        str(out_path),
    ]

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {in_path}")

    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frames_processed = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal = lut[gray]
            proc.stdin.write(thermal.tobytes())
            frames_processed += 1
    finally:
        cap.release()
        if proc.stdin:
            proc.stdin.close()
        proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {proc.returncode} for {out_path}")

    ow, oh, ofps, oframes, odur = get_video_info(out_path)
    assert ow == w and oh == h, f"Dimension mismatch: {ow}x{oh} vs {w}x{h}"
    assert oframes == total_frames, f"Frame count mismatch: {oframes} vs {total_frames}"
    assert abs(ofps - fps) < 0.5, f"FPS mismatch: {ofps} vs {fps}"
    assert out_path.exists() and out_path.stat().st_size > 0, "Output file missing or empty"

    return ow, oh, ofps, oframes, odur


def main():
    start_total = time.time()
    lut_path = Path("scripts/melobytes_medium_low_lut.npy")
    if not lut_path.exists():
        fallback = Path(r"C:\Users\omarb\.gemini\antigravity\brain\6406bae8-e501-4ff4-902c-8e1d0ed9bb1f\scratch\melobytes_medium_low_lut.npy")
        if fallback.exists():
            import shutil
            shutil.copy2(fallback, lut_path)
        else:
            raise FileNotFoundError(f"LUT file not found at {lut_path} or {fallback}")

    lut = np.load(str(lut_path))
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    raw_base = Path("data/raw")
    target_pairs = [
        (raw_base / "desert/negative/rgb", raw_base / "desert/negative/thermal"),
        (raw_base / "desert/positive/rgb", raw_base / "desert/positive/thermal"),
        (raw_base / "forest/negative/rgb", raw_base / "forest/negative/thermal"),
        (raw_base / "forest/positive/rgb", raw_base / "forest/positive/thermal"),
        (raw_base / "snow/negative/rgb", raw_base / "snow/negative/thermal"),
        (raw_base / "snow/positive/rgb", raw_base / "snow/positive/thermal"),
    ]

    all_jobs = []
    for in_dir, out_dir in target_pairs:
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("*.mp4"))
        for f in files:
            out_name = f.name.replace("RGB", "Thermal") if "RGB" in f.name else f.name.replace("rgb", "thermal")
            out_path = out_dir / out_name
            if out_path.exists() and out_path.stat().st_size > 0:
                continue
            all_jobs.append((f, out_path, in_dir.parent.parent.name, in_dir.parent.name))

    total = len(all_jobs)
    print("=" * 70)
    print(" BATCH CONVERTING RAW RGB SNIPPETS TO THERMAL VIA MELOBYTES LUT")
    print(f" Total Videos to Process: {total}")
    print("=" * 70)

    success = 0
    for idx, (in_file, out_file, biome, scenario) in enumerate(all_jobs, 1):
        t0 = time.time()
        print(f"[{idx:02d}/{total:02d}] ({biome}/{scenario}) {in_file.name} -> {out_file.name}...", end="", flush=True)
        ow, oh, ofps, oframes, odur = convert_video(in_file, out_file, lut, ffmpeg_exe)
        elapsed = time.time() - t0
        size_mb = out_file.stat().st_size / (1024 * 1024)
        speed = oframes / elapsed if elapsed > 0 else 0
        print(f" OK in {elapsed:.2f}s ({speed:.1f} fps) [{ow}x{oh}, {oframes}f, {size_mb:.2f} MB]")
        success += 1

    total_time = time.time() - start_total
    print("\n" + "=" * 70)
    print(f" CONVERSION COMPLETED: {success}/{total} videos converted in {total_time:.2f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()
