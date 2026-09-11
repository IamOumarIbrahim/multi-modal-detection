"""Median filter temporal post-processor."""

import statistics
from typing import Sequence
from mmsar.postprocessing.base import PostProcessor


class MedianFilterPostProcessor:
    """Trigger alarm when the median of the last 5 confidence values is >= tau.

    For frames where fewer than 5 frames have been observed (n < 5), only the
    frames available so far are used to calculate the median.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def decide(self, confidence: Sequence[float], tau: float) -> list[int]:
        """Convert confidence sequence into alarms based on rolling median."""
        decisions: list[int] = []
        for i in range(len(confidence)):
            start_idx = max(0, i - self.window_size + 1)
            window = confidence[start_idx : i + 1]
            med_val = statistics.median(window)
            decisions.append(1 if med_val >= tau else 0)
        return decisions
