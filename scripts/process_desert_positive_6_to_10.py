"""Move desert_RGB_positive_6 through 10 to videos/desert/positive/rgb and crop into dual 640x640 tiles in data/raw/desert/positive/rgb."""

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
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    duration = frames / fps if fps > 0 else 0
    return w, h, fps, frames, duration


def main():
    src_dir = Path(r"D:\Downloads\Videos")
    dest_videos = Path("videos/desert/positive/rgb")
    dest_raw = Path("data/raw/desert/positive/rgb")

    dest_videos.mkdir(parents=True, exist_ok=True)
    dest_raw.mkdir(parents=True, exist_ok=True)

    ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
    start_time = time.time()

    video_names = [f"desert_RGB_positive_{i}.mp4" for i in range(6, 11)]

    print("=" * 65)
    print("STEP 1: Moving raw videos from Downloads to repository videos dir")
    print("=" * 65)

    moved_videos = []
    for fname in video_names:
        src_path = src_dir / fname
        target_path = dest_videos / fname

        if target_path.exists():
            print(f"Target already exists in repo: {target_path} ({target_path.stat().st_size} bytes)")
            if src_path.exists():
                src_path.unlink()
                print(f"  Removed duplicate in source: {src_path}")
            moved_videos.append(target_path)
            continue

        if not src_path.exists():
            raise FileNotFoundError(f"Source video not found: {src_path}")

        src_size = src_path.stat().st_size
        print(f"Moving {fname} ({src_size} bytes)...")
        # Copy first with metadata
        shutil.copy2(src_path, target_path)
        assert target_path.exists() and target_path.stat().st_size == src_size, "Copy verification failed!"
        # Remove original from D:\Downloads\Videos
        src_path.unlink()
        print(f"  Successfully moved to {target_path}")
        moved_videos.append(target_path)

    print("\n" + "=" * 65)
    print("STEP 2: Processing videos into dual 640x640 tiles")
    print("=" * 65)

    total_tiles = 0
    for video_path in moved_videos:
        stem = video_path.stem
        w, h, fps, frames, dur = get_video_info(video_path)
        print(f"\nProcessing {video_path.name} ({w}x{h}, {fps:.1f} fps, {frames} frames, {dur:.2f}s):")

        assert w == 1280 and h == 720, f"Expected 1280x720, got {w}x{h} for {video_path}"

        crops = [
            ("left", "crop=640:640:0:40", dest_raw / f"{stem}_left.mp4"),
            ("right", "crop=640:640:640:40", dest_raw / f"{stem}_right.mp4"),
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
            assert oframes == 240, f"Invalid frame count: {oframes}"
            assert abs(ofps - 24.0) < 0.1, f"Invalid FPS: {ofps}"

            total_tiles += 1

    elapsed = time.time() - start_time
    print("\n" + "=" * 65)
    print(" SUMMARY")
    print("=" * 65)
    print(f"Videos moved:     {len(moved_videos)}")
    print(f"Tiles generated:  {total_tiles}")
    print(f"Elapsed Time:     {elapsed:.2f}s")
    print("SUCCESS: All videos moved and tiled successfully!")


if __name__ == "__main__":
    main()
