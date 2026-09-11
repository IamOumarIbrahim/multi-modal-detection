"""History tracking post-processor using temporal window count."""

from typing import Sequence
from mmsar.postprocessing.base import PostProcessor


class HistoryTrackingPostProcessor:
    """Trigger alarm when >= 3 of the last 5 confidence values are >= tau.

    For frames where fewer than 5 frames have been observed (n < 5), only the
    frames available so far are considered in the window.
    """

    def __init__(self, window_size: int = 5, required_hits: int = 3):
        self.window_size = window_size
        self.required_hits = required_hits

    def decide(self, confidence: Sequence[float], tau: float) -> list[int]:
        """Convert confidence sequence into alarms based on temporal hit count."""
        decisions: list[int] = []
        for i in range(len(confidence)):
            start_idx = max(0, i - self.window_size + 1)
            window = confidence[start_idx : i + 1]
            hits = sum(1 for c in window if c >= tau)
            decisions.append(1 if hits >= self.required_hits else 0)
        return decisions
