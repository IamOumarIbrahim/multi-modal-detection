"""Video I/O utilities for decimation and spatial cropping."""

from pathlib import Path
from typing import Union, Sequence
import cv2


def decimate_and_crop(
    video_path: Union[str, Path],
    out_dir: Union[str, Path],
    decimation_ratio: int = 3,
    crop_x: tuple[int, int] = (320, 960),
    crop_y: tuple[int, int] = (80, 720),
) -> list[Path]:
    """Decimate a video (e.g. 24 -> 8 fps via 3:1 decimation) and spatially crop frames to 640x640.

    Args:
        video_path: Path to the input video file.
        out_dir: Directory where extracted frames will be saved.
        decimation_ratio: Step ratio for temporal decimation (default 3:1).
        crop_x: (x_min, x_max) pixel coordinates (default: 320 to 960).
        crop_y: (y_min, y_max) pixel coordinates (default: 80 to 720).

    Returns:
        List of Paths to the saved cropped frame images in sequential order.

    Raises:
        FileNotFoundError: If video file does not exist.
        RuntimeError: If video cannot be opened by OpenCV.
    """
    in_path = Path(video_path)
    if not in_path.exists():
        raise FileNotFoundError(f"Video file not found at {in_path}")

    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(in_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video at {in_path}")

    saved_frames: list[Path] = []
    frame_idx = 0
    saved_idx = 0

    x_min, x_max = crop_x
    y_min, y_max = crop_y

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % decimation_ratio == 0:
                # Crop [Y_min:Y_max, X_min:X_max]
                cropped = frame[y_min:y_max, x_min:x_max]

                out_filename = destination / f"frame_{saved_idx:06d}.png"
                success = cv2.imwrite(str(out_filename), cropped)
                if not success:
                    raise IOError(f"Failed to write image frame to {out_filename}")

                saved_frames.append(out_filename)
                saved_idx += 1

            frame_idx += 1
    finally:
        cap.release()

    return saved_frames
