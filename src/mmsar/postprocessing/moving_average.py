"""Moving average temporal filter post-processor."""

from typing import Sequence
from mmsar.postprocessing.base import PostProcessor


class MovingAveragePostProcessor:
    """Trigger alarm when the mean of the last 5 confidence values is >= tau.

    For frames where fewer than 5 frames have been observed (n < 5), only the
    frames available so far are used to calculate the mean.
    """

    def __init__(self, window_size: int = 5):
        self.window_size = window_size

    def decide(self, confidence: Sequence[float], tau: float) -> list[int]:
        """Convert confidence sequence into alarms based on moving average."""
        decisions: list[int] = []
        for i in range(len(confidence)):
            start_idx = max(0, i - self.window_size + 1)
            window = confidence[start_idx : i + 1]
            mean_val = sum(window) / len(window)
            decisions.append(1 if mean_val >= tau else 0)
        return decisions
