"""Tests for video decimation, cropping, and frame sampling."""

from pathlib import Path
import cv2
import numpy as np
import pytest
from tests.fixtures.make_synthetic_video import create_synthetic_video
from mmsar.preprocessing.video_io import decimate_and_crop
from mmsar.preprocessing.frame_sampler import sample_frames_from_manifest


def test_decimate_and_crop_synthetic_fixture(tmp_path: Path) -> None:
    # 1. Create 2s, 24fps, 1280x720 video fixture (48 frames total)
    video_path = tmp_path / "fixture_video.mp4"
    create_synthetic_video(video_path, duration_sec=2, fps=24, width=1280, height=720)
    assert video_path.exists(), "Synthetic fixture video was not created"

    # 2. Run decimate_and_crop (3:1 decimation, crop to 640x640)
    out_dir = tmp_path / "cropped_frames"
    saved_frames = decimate_and_crop(video_path, out_dir, decimation_ratio=3)

    # 3. Assert exactly 16 frames are produced (2s * 8fps = 16)
    assert len(saved_frames) == 16, f"Expected exactly 16 frames, but got {len(saved_frames)}"

    # 4. Assert each frame has dimensions exactly 640x640
    for frame_path in saved_frames:
        img = cv2.imread(str(frame_path))
        assert img is not None, f"Failed to read image at {frame_path}"
        assert img.shape == (640, 640, 3), (
            f"Expected frame shape (640, 640, 3), but got {img.shape} for {frame_path.name}"
        )

    # 5. Check correct temporal ordering via the drawn index counter
    # In make_synthetic_video, binary blocks are at original X: [350 + b*40], Y: [150]
    # In cropped frame [80:720, 320:960]:
    # Relative X: (350 - 320) + b*40 = 30 + b*40
    # Relative Y: 150 - 80 = 70
    extracted_indices: list[int] = []
    for frame_path in saved_frames:
        img = cv2.imread(str(frame_path))
        recovered_idx = 0
        for b in range(6):
            rx = 30 + b * 40
            ry = 70
            patch = img[ry + 5 : ry + 25, rx + 5 : rx + 25]
            if np.mean(patch) > 128:
                recovered_idx |= (1 << b)
        extracted_indices.append(recovered_idx)

    expected_indices = list(range(0, 48, 3))  # [0, 3, 6, 9, 12, ..., 45] -> 16 frames
    assert extracted_indices == expected_indices, (
        f"Frame ordering mismatch:\nExpected: {expected_indices}\nGot: {extracted_indices}"
    )


def test_frame_sampler_synthetic(tmp_path: Path) -> None:
    source_dir = tmp_path / "mock_raw"
    dest_dir = tmp_path / "mock_processed"

    vid1 = source_dir / "cond_a" / "scene_1" / "video1.mp4"
    create_synthetic_video(vid1, duration_sec=1, fps=24)

    summary = sample_frames_from_manifest(source_dir, dest_dir, decimation_ratio=3)
    assert summary["processed_videos"] == 1
    assert summary["total_frames"] == 8  # 1s * 8fps = 8 frames
