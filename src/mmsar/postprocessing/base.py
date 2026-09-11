"""Abstract Base Classes and Protocols for post-processing methods."""

from typing import Protocol, Sequence, runtime_checkable


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for post-processing confidence sequences into binary alarm decisions."""

    def decide(self, confidence: Sequence[float], tau: float) -> list[int]:
        """Convert a sequence of per-frame confidence scores into binary alarm decisions.

        Args:
            confidence: Sequence of frame-level confidence scores in [0.0, 1.0].
            tau: Alarm decision threshold in [0.0, 1.0].

        Returns:
            List of binary alarm decisions (0: no alarm, 1: alarm) for each frame.
        """
        ...
