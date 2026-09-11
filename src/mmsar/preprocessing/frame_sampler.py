"""Frame sampling pipeline for processing raw dataset videos into cropped frames."""

from pathlib import Path
from typing import Union, Optional
from mmsar.manifest.store import load_manifest, DEFAULT_MANIFEST_PATH
from mmsar.preprocessing.video_io import decimate_and_crop


def sample_frames_from_manifest(
    source_dir: Union[str, Path],
    dest_dir: Union[str, Path],
    manifest_path: Optional[Union[str, Path]] = None,
    decimation_ratio: int = 3,
) -> dict[str, int]:
    """Scan raw video directory for eligible videos and export decimated cropped frames.

    Args:
        source_dir: Root directory containing raw videos (e.g. condition/scenario folders).
        dest_dir: Target directory where cropped frames will be saved.
        manifest_path: Optional path to manifest.json to check target quotas.
        decimation_ratio: Step ratio for frame extraction (default 3:1 for 24->8 fps).

    Returns:
        Summary dict containing counts of processed videos and extracted frames.
    """
    src = Path(source_dir)
    dst = Path(dest_dir)
    dst.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest(manifest_path) if manifest_path else None

    # Search for all .mp4 and .mov files in source_dir recursively
    video_extensions = ("*.mp4", "*.mov")
    video_files: list[Path] = []
    for ext in video_extensions:
        video_files.extend(src.rglob(ext))

    processed_videos = 0
    total_frames = 0

    for video_file in sorted(video_files):
        # Determine relative folder structure or video name
        rel_path = video_file.relative_to(src)
        video_stem = video_file.stem

        video_out_dir = dst / rel_path.parent / video_stem
        frames = decimate_and_crop(
            video_path=video_file,
            out_dir=video_out_dir,
            decimation_ratio=decimation_ratio,
        )

        processed_videos += 1
        total_frames += len(frames)

    return {
        "processed_videos": processed_videos,
        "total_frames": total_frames,
    }
