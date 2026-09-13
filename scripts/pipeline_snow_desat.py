"""Convert Snow RGB video snippets to 100% Desaturated Pseudo-Thermal (pipeline_snow_desat).

Applies the original Melobytes thermal LUT pipeline, followed immediately by 100%
desaturation (converting the colored thermal output to monochrome/white-hot grayscale).

This eliminates the saturated red-out over bright snow surfaces, rendering snow
as a dark, cool terrain while highlighting warm targets and terrain contours.

Usage:
  python scripts/pipeline_snow_desat.py <input_rgb.mp4> <output_thermal.mp4>
  python scripts/pipeline_snow_desat.py  # Batch processes data/raw/snow/
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


def build_desaturated_lut(lut: np.ndarray) -> np.ndarray:
    """Precompute a 256x3 LUT that applies Melobytes LUT + 100% desaturation."""
    # lut is (256, 3) in BGR
    melo_bgr = lut.reshape((256, 1, 3))
    melo_gray = cv2.cvtColor(melo_bgr, cv2.COLOR_BGR2GRAY).squeeze()
    # Expand to 3-channel BGR for FFmpeg bgr24 pipe
    return np.repeat(melo_gray[:, None], 3, axis=1).astype(np.uint8)


def convert_video_desat(in_path: Path, out_path: Path, desat_lut: np.ndarray, ffmpeg_exe: str):
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
            thermal_desat = desat_lut[gray]
            proc.stdin.write(thermal_desat.tobytes())
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
    desat_lut = build_desaturated_lut(lut)
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    # CLI mode
    if len(sys.argv) >= 3:
        in_p = Path(sys.argv[1])
        out_p = Path(sys.argv[2])
        out_p.parent.mkdir(parents=True, exist_ok=True)
        print(f"Converting {in_p.name} -> {out_p.name} (pipeline_snow_desat)...", flush=True)
        ow, oh, ofps, oframes, odur = convert_video_desat(in_p, out_p, desat_lut, ffmpeg_exe)
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
    print(" BATCH CONVERTING SNOW RGB VIDEOS (pipeline_snow_desat)", flush=True)
    print(f" Total Videos: {len(all_jobs)}", flush=True)
    print("=" * 70, flush=True)

    for idx, (in_file, out_file) in enumerate(all_jobs, 1):
        t0 = time.time()
        print(f"[{idx:02d}/{len(all_jobs):02d}] {in_file.name} -> {out_file.name}...", end="", flush=True)
        convert_video_desat(in_file, out_file, desat_lut, ffmpeg_exe)
        print(f" OK in {time.time() - t0:.2f}s", flush=True)

    print(f"\nAll snow conversions finished in {time.time() - start_total:.2f}s.", flush=True)


if __name__ == "__main__":
    main()
