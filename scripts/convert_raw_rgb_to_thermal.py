#   MMSAR - step 2: make the pseudo-thermal (pseudo-TIR) twin of every RGB tile
#        Oumar Ibrahim U22200741

# each frame: gray = 0.299 R + 0.587 G + 0.114 B, then the 256 level lookup table
# scripts/melobytes_medium_low_lut.npy maps every gray level to a thermal colour
# input   data/raw/{biome}/{scenario}/rgb/*RGB*.mp4
# output  data/raw/{biome}/{scenario}/thermal/*Thermal*.mp4 (h264, crf 18, same fps)
# clips that already have a thermal file are skipped

import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np


def convert_video(in_path, out_path, lut, ffmpeg):
    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # raw bgr frames go into ffmpeg through stdin
    cmd = [
        ffmpeg, "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo",
        "-s", f"{w}x{h}", "-pix_fmt", "bgr24", "-r", str(fps),
        "-i", "-",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "fast", "-crf", "18",
        "-movflags", "+faststart",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)

    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal = lut[gray]
        proc.stdin.write(thermal.tobytes())
        count += 1
    cap.release()
    proc.stdin.close()
    proc.wait()

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg failed for {out_path}")
    if count != total:
        raise RuntimeError(f"{in_path}: read {count} frames, expected {total}")
    return count


def main():
    lut = np.load("scripts/melobytes_medium_low_lut.npy")
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()

    jobs = []
    for rgb_path in sorted(Path("data/raw").glob("*/*/rgb/*.mp4")):
        out_dir = rgb_path.parent.parent / "thermal"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / rgb_path.name.replace("RGB", "Thermal")
        if out_path.exists() and out_path.stat().st_size > 0:
            continue
        jobs.append((rgb_path, out_path))

    print("clips to convert:", len(jobs))
    for i in range(len(jobs)):
        rgb_path, out_path = jobs[i]
        n = convert_video(rgb_path, out_path, lut, ffmpeg)
        print(f"[{i + 1}/{len(jobs)}] {rgb_path.name} -> {out_path.name} ({n} frames)")

    print()
    print("----------------------")
    print("Converted= ", len(jobs))
    print("----------------------")


if __name__ == "__main__":
    main()
