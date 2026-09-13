"""Convert Snow RGB video snippets to Pseudo-Thermal by inverting the Melobytes LUT mapping.

In the standard pipeline (Desert/Forest):
- gray = 255 -> lut[255] (Red/Yellow = Hot)
- gray = 0   -> lut[0]   (Blue/Purple = Cold)

In the Snow pipeline (pipeline_snow):
- Snow is high-albedo/bright in visible light, but physically sub-zero/cold.
- Inverting the mapping (255 - gray) maps:
  - Bright white snow (gray ~ 240-255) -> lut[0-15] (Cold Blue/Cyan/Purple)
  - Dark targets, hikers, rocks, shadows (gray ~ 40-80) -> lut[175-215] (Warm/Hot Red/Orange/Yellow)

Usage:
  python scripts/pipeline_snow.py <input_rgb.mp4> <output_thermal.mp4>
  python scripts/pipeline_snow.py  # Batch processes data/raw/snow/
"""

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


def convert_snow_video(in_path: Path, out_path: Path, lut: np.ndarray, ffmpeg_exe: str):
    """Convert an RGB snow video to pseudo-thermal using inverted grayscale mapping."""
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

    # Invert LUT mapping for snow: bright snow -> cold blue, dark objects -> warm red
    inv_lut = lut[::-1]

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            thermal = inv_lut[gray]
            proc.stdin.write(thermal.tobytes())
            frames_processed += 1
    finally:
        cap.release()
        if proc.stdin:
            try:
                proc.stdin.close()
            except Exception:
                pass
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
            raise FileNotFoundError(f"LUT file not found at {lut_path}")

    lut = np.load(str(lut_path))
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # CLI mode: python scripts/pipeline_snow.py input.mp4 output.mp4
    if len(sys.argv) >= 3:
        in_p = Path(sys.argv[1])
        out_p = Path(sys.argv[2])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting {in_p.name} -> {out_p.name} (pipeline_snow)...", flush=True)
        ow, oh, ofps, oframes, odur = convert_snow_video(in_p, out_p, lut, ffmpeg_exe)
        print(f"Done in {time.time() - start_total:.2f}s [{ow}x{oh}, {oframes} frames, {out_p.stat().st_size / (1024*1024):.2f} MB]", flush=True)
        return

    # Batch directory mode
    raw_base = Path("data/raw/snow")
    target_pairs = [
        (raw_base / "negative/rgb", raw_base / "negative/thermal"),
        (raw_base / "positive/rgb", raw_base / "positive/thermal"),
    ]

    all_jobs = []
    for in_dir, out_dir in target_pairs:
        if not in_dir.exists():
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        files = sorted(in_dir.glob("*.mp4"))
        for f in files:
            out_name = f.name.replace("RGB", "Thermal") if "RGB" in f.name else f.name.replace("rgb", "thermal")
            all_jobs.append((f, out_dir / out_name))

    if not all_jobs:
        print("No snow videos found in data/raw/snow/ to process.", flush=True)
        return

    print("=" * 70, flush=True)
    print(" BATCH CONVERTING SNOW RGB VIDEOS TO INVERTED THERMAL (pipeline_snow)", flush=True)
    print(f" Total Videos: {len(all_jobs)}", flush=True)
    print("=" * 70, flush=True)

    for idx, (in_file, out_file) in enumerate(all_jobs, 1):
        t0 = time.time()
        print(f"[{idx:02d}/{len(all_jobs):02d}] {in_file.name} -> {out_file.name}...", end="", flush=True)
        convert_snow_video(in_file, out_file, lut, ffmpeg_exe)
        print(f" OK in {time.time() - t0:.2f}s", flush=True)

    print(f"\nAll snow conversions finished in {time.time() - start_total:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
