"""Baseline per-frame thresholding post-processor."""

from typing import Sequence
from mmsar.postprocessing.base import PostProcessor


class BaselinePostProcessor:
    """Evaluates alarm decisions frame-by-frame at threshold tau without temporal history."""

    def decide(self, confidence: Sequence[float], tau: float) -> list[int]:
        """Produce alarm decisions: 1 if confidence[n] >= tau else 0."""
        return [1 if c >= tau else 0 for c in confidence]
