"""Move uncopied Snow RGB videos from D:\Downloads, dual-crop to 640x640, and save to data/raw/snow.

Processes:
- Positives:
    - Snow_RGB_positive_9  -> snow_RGB_positive_9_left/right.mp4
    - Snow_RGB_positive_10 -> snow_RGB_positive_10_left/right.mp4
    - Snow_RGB_positive_11 -> snow_RGB_positive_11_left/right.mp4
    - Snow_RGB_positive_12 -> snow_RGB_positive_12_left/right.mp4
    - Snow_RGB_positive_13 -> snow_RGB_positive_13_left/right.mp4
    - Snow_RGB_positive_14 -> snow_RGB_positive_14_left/right.mp4
    - Create_approximately_second.mp4 -> snow_RGB_positive_15_left/right.mp4
- Negatives:
    - Snow_RGB_hard_negative_1 -> snow_RGB_negative_1_left/right.mp4

Also moves existing Snow_RGB_positive_1..8 raw files to videos/snow/positive/rgb for completeness.
"""

from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path
import cv2
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


def main():
    start_time = time.time()
    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()

    downloads_dir = Path(r"D:\Downloads")
    downloads_videos_archive = Path(r"D:\Downloads\Videos")
    repo_videos_base = Path("videos/snow")
    repo_raw_base = Path("data/raw/snow")

    downloads_videos_archive.mkdir(parents=True, exist_ok=True)
    (repo_videos_base / "positive/rgb").mkdir(parents=True, exist_ok=True)
    (repo_videos_base / "negative/rgb").mkdir(parents=True, exist_ok=True)
    (repo_raw_base / "positive/rgb").mkdir(parents=True, exist_ok=True)
    (repo_raw_base / "negative/rgb").mkdir(parents=True, exist_ok=True)

    jobs = [
        ("Snow_RGB_positive_9.mp4", "snow_RGB_positive_9", "positive"),
        ("Snow_RGB_positive_10.mp4", "snow_RGB_positive_10", "positive"),
        ("Snow_RGB_positive_11.mp4", "snow_RGB_positive_11", "positive"),
        ("Snow_RGB_positive_12.mp4", "snow_RGB_positive_12", "positive"),
        ("Snow_RGB_positive_13.mp4", "snow_RGB_positive_13", "positive"),
        ("Snow_RGB_positive_14.mp4", "snow_RGB_positive_14", "positive"),
        ("Create_approximately_second.mp4", "snow_RGB_positive_15", "positive"),
        ("Snow_RGB_hard_negative_1.mp4", "snow_RGB_negative_1", "negative"),
    ]

    print("=" * 70)
    print(" STEP 1: MOVING RAW SOURCE VIDEOS TO videos/snow & ARCHIVE")
    print("=" * 70)

    moved_jobs = []
    for src_name, canonical_stem, scenario in jobs:
        src_path = downloads_dir / src_name
        dest_video_dir = repo_videos_base / scenario / "rgb"
        dest_canonical_name = f"{canonical_stem}.mp4"
        dest_video_path = dest_video_dir / dest_canonical_name
        archive_path = downloads_videos_archive / dest_canonical_name

        if not src_path.exists():
            if dest_video_path.exists():
                print(f"[ALREADY MOVED] {dest_video_path}")
                moved_jobs.append((dest_video_path, canonical_stem, scenario))
                continue
            raise FileNotFoundError(f"Source video {src_path} does not exist!")

        src_size = src_path.stat().st_size
        print(f"Moving: {src_name} -> {dest_video_path.name} ({src_size / (1024*1024):.2f} MB)")

        # 1. Archive to D:\Downloads\Videos
        shutil.copy2(src_path, archive_path)
        assert archive_path.exists() and archive_path.stat().st_size == src_size

        # 2. Copy to repository videos directory
        shutil.copy2(src_path, dest_video_path)
        assert dest_video_path.exists() and dest_video_path.stat().st_size == src_size

        # 3. Unlink from D:\Downloads
        src_path.unlink()
        print(f"  [MOVED] Verified in {dest_video_path} and unlinked from {downloads_dir}")
        moved_jobs.append((dest_video_path, canonical_stem, scenario))

    # Also move previously processed Snow_RGB_positive_1..8 to videos/snow/positive/rgb and archive
    print("\nMoving previously harvested Snow 1..8 source videos to repo & archive...")
    for i in range(1, 9):
        p_name = f"Snow_RGB_positive_{i}.mp4"
        p_src = downloads_dir / p_name
        if p_src.exists():
            p_dst = repo_videos_base / "positive/rgb" / f"snow_RGB_positive_{i}.mp4"
            p_arch = downloads_videos_archive / f"snow_RGB_positive_{i}.mp4"
            shutil.copy2(p_src, p_arch)
            shutil.copy2(p_src, p_dst)
            p_src.unlink()
            print(f"  Moved {p_name} -> {p_dst.name}")

    print("\n" + "=" * 70)
    print(" STEP 2: CROPPING VIDEOS INTO DUAL 640x640 TILES")
    print("=" * 70)

    generated_tiles = []
    for video_path, canonical_stem, scenario in moved_jobs:
        w, h, fps, frames, dur = get_video_info(video_path)
        print(f"\nProcessing {video_path.name} ({w}x{h}, {fps:.1f} fps, {frames} frames, {dur:.2f}s):")
        assert w == 1280 and h == 720, f"Expected 1280x720, got {w}x{h}"

        dest_raw_dir = repo_raw_base / scenario / "rgb"
        crops = [
            ("left", "crop=640:640:0:40", dest_raw_dir / f"{canonical_stem}_left.mp4"),
            ("right", "crop=640:640:640:40", dest_raw_dir / f"{canonical_stem}_right.mp4"),
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
            sz_mb = out_path.stat().st_size / (1024 * 1024)
            print(
                f"  [{side.upper():<5}] -> {out_path.name:<32} "
                f"({ow}x{oh}, {ofps:.1f} fps, {oframes} frames, {sz_mb:.2f} MB)"
            )

            assert ow == 640 and oh == 640, f"Invalid dimensions: {ow}x{oh}"
            assert oframes == frames, f"Frame count mismatch: {oframes} vs {frames}"
            assert abs(ofps - 24.0) < 0.1, f"Invalid FPS: {ofps}"
            assert out_path.exists() and out_path.stat().st_size > 0, "Output missing or empty"
            generated_tiles.append(out_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 70)
    print(" HARVESTING SUMMARY")
    print("=" * 70)
    print(f"Videos Processed: {len(moved_jobs)}")
    print(f"Tiles Generated:  {len(generated_tiles)} (expected {len(moved_jobs) * 2})")
    print(f"Elapsed Time:     {elapsed:.2f}s")
    print("All snow videos successfully processed and saved into data/raw/snow!")


if __name__ == "__main__":
    main()
