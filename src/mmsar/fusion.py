"""Late fusion module for combining multi-modal detection confidence scores."""

from typing import Sequence


def late_fusion(
    c_rgb: Sequence[float],
    c_thermal: Sequence[float],
) -> list[float]:
    """Compute late fusion of RGB and Thermal confidence scores using max-fusion.

    Formula: s[n] = max(c_rgb[n], c_thermal[n])

    Args:
        c_rgb: Sequence of frame-level confidence scores from the RGB detector.
        c_thermal: Sequence of frame-level confidence scores from the Thermal detector.

    Returns:
        List of fused confidence scores per frame.

    Raises:
        ValueError: If c_rgb and c_thermal have differing lengths.
    """
    if len(c_rgb) != len(c_thermal):
        raise ValueError(
            f"Stream length mismatch: c_rgb has length {len(c_rgb)} "
            f"but c_thermal has length {len(c_thermal)}"
        )
    return [max(r, t) for r, t in zip(c_rgb, c_thermal)]
