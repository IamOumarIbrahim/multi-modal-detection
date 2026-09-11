"""Annotation tooling for Label Studio integration and cross-modality labeling."""

from mmsar.annotation.label_studio_config import (
    generate_label_studio_config,
    validate_label_studio_config,
    DEFAULT_LABEL_NAME,
)
from mmsar.annotation.label_studio_client import LabelStudioManager
from mmsar.annotation.rgb_to_thermal_copy import copy_annotations_rgb_to_thermal

__all__ = [
    "generate_label_studio_config",
    "validate_label_studio_config",
    "DEFAULT_LABEL_NAME",
    "LabelStudioManager",
    "copy_annotations_rgb_to_thermal",
]
