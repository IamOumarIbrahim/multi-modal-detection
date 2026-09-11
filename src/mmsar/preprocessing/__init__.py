"""Video preprocessing and frame decimation package."""

from mmsar.preprocessing.video_io import decimate_and_crop
from mmsar.preprocessing.frame_sampler import sample_frames_from_manifest

__all__ = [
    "decimate_and_crop",
    "sample_frames_from_manifest",
]
