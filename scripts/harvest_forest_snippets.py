"""Harvest dual 640x640 snippets (left and right) from forest RGB videos in Downloads.

Processes:
- Positives: forest_RGB_positive_1 to 6 (1280x720 -> 2x 640x640, 240 frames)
             forest_RGB_positive_7 (752x416 -> scaled to 1280x720 -> 2x 640x640, 145 frames)
- Negatives: forest_RGB_hard_negative_1 to 5 (752x416 -> scaled to 1280x720 -> 2x 640x640, 145 frames)

Outputs to:
- data/raw/forest/positive/rgb/
- data/raw/forest/negative/rgb/
"""

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


def harvest_forest_snippets(
    source_dir: Path = Path(r"D:\Downloads\Videos"),
    dest_base: Path = Path("data/raw/forest"),
):
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    start_time = time.time()
    total_harvested = 0

    print("=" * 65)
    print(" STARTING FOREST DUAL-CROP HARVESTING PIPELINE")
    print("=" * 65)

    tasks = [
        ("positive", [f"forest_RGB_positive_{i}.mp4" for i in range(1, 8)]),
        ("negative", [f"forest_RGB_hard_negative_{i}.mp4" for i in range(1, 6)]),
    ]

    for scenario, file_names in tasks:
        out_dir = dest_base / scenario / "rgb"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nProcessing scenario: '{scenario}' -> Output: {out_dir}")

        for fname in file_names:
            video_path = source_dir / fname
            if not video_path.exists():
                print(f"  [SKIPPED] {fname} not found in {source_dir}")
                continue

            stem = video_path.stem
            w, h, fps, frames, dur = get_video_info(video_path)
            print(f"\nSource Video: {fname} ({w}x{h}, {fps:.1f} fps, {frames} frames, {dur:.2f}s)")

            # Determine filter: if already 1280x720, crop directly; otherwise scale with lanczos first
            if w == 1280 and h == 720:
                left_filter = "crop=640:640:0:40"
                right_filter = "crop=640:640:640:40"
            else:
                print(f"  Scaling {w}x{h} -> 1280x720 (lanczos) before dual-crop...")
                left_filter = "scale=1280:720:flags=lanczos,crop=640:640:0:40"
                right_filter = "scale=1280:720:flags=lanczos,crop=640:640:640:40"

            crops = [
                ("left", left_filter, out_dir / f"{stem}_left.mp4"),
                ("right", right_filter, out_dir / f"{stem}_right.mp4"),
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
                print(
                    f"  [{side.upper()}] -> {out_path.name}: "
                    f"{ow}x{oh}, {ofps:.1f} fps, {oframes} frames, {odur:.2f}s ({size_mb:.2f} MB)"
                )

                assert ow == 640 and oh == 640, f"Invalid dimensions: {ow}x{oh}"
                assert abs(ofps - 24.0) < 0.1, f"Invalid FPS: {ofps}"
                assert oframes == frames, f"Frame count mismatch: {oframes} vs {frames}"

                total_harvested += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(f" COMPLETED: Harvested {total_harvested} snippets in {elapsed:.2f} seconds.")
    print("=" * 65)


if __name__ == "__main__":
    harvest_forest_snippets()
