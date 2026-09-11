"""Fixture generator for synthetic video files used in testing."""

from pathlib import Path
from typing import Union
import cv2
import numpy as np


def create_synthetic_video(
    output_path: Union[str, Path],
    duration_sec: int = 2,
    fps: int = 24,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Create a synthetic video with index counter encoded on each frame.

    Args:
        output_path: Destination path for the synthetic .mp4 file.
        duration_sec: Length of video in seconds (default 2s).
        fps: Frames per second (default 24).
        width: Frame width (default 1280).
        height: Frame height (default 720).

    Returns:
        Path to the written video.
    """
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, float(fps), (width, height))

    total_frames = duration_sec * fps  # 48 frames for 2s @ 24fps

    for idx in range(total_frames):
        # Create background
        frame = np.full((height, width, 3), 40, dtype=np.uint8)

        # Draw frame number as text inside the future crop box X[320, 960], Y[80, 720]
        text = f"FRAME_{idx:04d}"
        cv2.putText(
            frame,
            text,
            (400, 300),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.5,
            (255, 255, 255),
            3,
            cv2.LINE_AA,
        )

        # Encode frame index using binary square blocks resistant to lossy video compression
        # Located at X in [350, 600], Y in [150, 180] (strictly inside crop box X[320, 960], Y[80, 720])
        for b in range(6):
            bit_val = 255 if (idx & (1 << b)) else 0
            x1 = 350 + b * 40
            y1 = 150
            cv2.rectangle(frame, (x1, y1), (x1 + 30, y1 + 30), (bit_val, bit_val, bit_val), -1)

        writer.write(frame)

    writer.release()
    return out_path
