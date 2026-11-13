#   MMSAR - step 1: cut every 1280x720 source video into two 640x640 tiles
#        Oumar Ibrahim U22200741

# input   videos/{biome}/{scenario}/rgb/*.mp4
# output  data/raw/{biome}/{scenario}/rgb/{stem}_left.mp4 and {stem}_right.mp4
# left tile x = 0, right tile x = 640, both at y = 40 (centered vertically)
# sources that are not 1280x720 are scaled up first (lanczos)

import subprocess
import sys
from pathlib import Path

import cv2
import imageio_ffmpeg


def video_info(path):
    cap = cv2.VideoCapture(str(path))
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return w, h, fps, frames


def main():
    src_root = Path("videos")
    dst_root = Path("data/raw")
    if len(sys.argv) > 1:
        src_root = Path(sys.argv[1])

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    n_videos = 0
    n_tiles = 0

    for video in sorted(src_root.glob("*/*/rgb/*.mp4")):
        scenario = video.parent.parent.name
        biome = video.parent.parent.parent.name
        out_dir = dst_root / biome / scenario / "rgb"
        out_dir.mkdir(parents=True, exist_ok=True)

        w, h, fps, frames = video_info(video)
        print(f"{biome}/{scenario}  {video.name}  {w}x{h}  {frames} frames")

        scale = ""
        if w != 1280 or h != 720:
            scale = "scale=1280:720:flags=lanczos,"

        for side, x in [("left", 0), ("right", 640)]:
            out_path = out_dir / f"{video.stem}_{side}.mp4"
            cmd = [
                ffmpeg, "-y", "-i", str(video),
                "-vf", f"{scale}crop=640:640:{x}:40",
                "-c:v", "libx264", "-crf", "18", "-preset", "fast",
                "-pix_fmt", "yuv420p", "-r", "24",
                "-movflags", "+faststart",
                str(out_path),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"ffmpeg failed on {video}:\n{res.stderr}")

            # every tile must keep the size, the frame rate and all the frames
            ow, oh, ofps, oframes = video_info(out_path)
            if ow != 640 or oh != 640 or abs(ofps - 24.0) > 0.1 or oframes != frames:
                raise RuntimeError(f"bad tile {out_path}: {ow}x{oh} {ofps} fps {oframes} frames")
            print(f"    -> {out_path.name}")
            n_tiles += 1

        n_videos += 1

    print()
    print("----------------------")
    print("Source videos= ", n_videos)
    print("Tiles written= ", n_tiles)
    print("----------------------")


if __name__ == "__main__":
    main()
