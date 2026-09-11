"""Tests for late fusion gate."""

import pytest
from mmsar.fusion import late_fusion


def test_late_fusion_hand_computed() -> None:
    # 5 frames sequence:
    # Frame 0: RGB=0.2, Thermal=0.7 -> Thermal beats RGB -> fused = 0.7
    # Frame 1: RGB=0.8, Thermal=0.3 -> RGB beats Thermal -> fused = 0.8
    # Frame 2: RGB=0.4, Thermal=0.4 -> Equal -> fused = 0.4
    # Frame 3: RGB=0.0, Thermal=0.9 -> Thermal beats RGB -> fused = 0.9
    # Frame 4: RGB=0.95, Thermal=0.1 -> RGB beats Thermal -> fused = 0.95
    c_rgb = [0.2, 0.8, 0.4, 0.0, 0.95]
    c_thermal = [0.7, 0.3, 0.4, 0.9, 0.1]
    expected_fused = [0.7, 0.8, 0.4, 0.9, 0.95]

    fused = late_fusion(c_rgb, c_thermal)
    assert fused == expected_fused


def test_late_fusion_length_mismatch() -> None:
    c_rgb = [0.2, 0.8]
    c_thermal = [0.7]
    with pytest.raises(ValueError, match="Stream length mismatch"):
        late_fusion(c_rgb, c_thermal)


def test_late_fusion_empty() -> None:
    assert late_fusion([], []) == []
