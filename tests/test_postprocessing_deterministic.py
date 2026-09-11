"""Tests for deterministic post-processing methods."""

import pytest
from mmsar.postprocessing.base import PostProcessor
from mmsar.postprocessing.baseline import BaselinePostProcessor
from mmsar.postprocessing.history_tracking import HistoryTrackingPostProcessor
from mmsar.postprocessing.moving_average import MovingAveragePostProcessor
from mmsar.postprocessing.median_filter import MedianFilterPostProcessor


# Chosen 10-frame confidence sequence with tau = 0.5:
# Frame 0: 0.1 -> window=[0.1]
#   Base: 0, Hist(>=0.5): 0, MA(mean=0.1): 0, Med(0.1): 0
# Frame 1: 0.2 -> window=[0.1, 0.2]
#   Base: 0, Hist: 0, MA(0.15): 0, Med(0.15): 0
# Frame 2: 0.8 -> window=[0.1, 0.2, 0.8] -- SINGLE FRAME SPIKE
#   Base: 1 (accepted)
#   Hist: 1 (count=1 < 3 -> 0)
#   MA: (1.1/3 ~ 0.3667 < 0.5 -> 0)
#   Med: median([0.1, 0.2, 0.8]) = 0.2 < 0.5 -> 0 (correctly rejected)
# Frame 3: 0.1 -> window=[0.1, 0.2, 0.8, 0.1]
#   Base: 0, Hist: 1 -> 0, MA(1.2/4 = 0.30): 0, Med(0.15): 0
# Frame 4: 0.2 -> window=[0.1, 0.2, 0.8, 0.1, 0.2]
#   Base: 0, Hist: 1 -> 0, MA(1.4/5 = 0.28): 0, Med(0.2): 0
# Frame 5: 0.6 -> window=[0.2, 0.8, 0.1, 0.2, 0.6]
#   Base: 1, Hist: 2 -> 0, MA(1.9/5 = 0.38): 0, Med(0.2): 0
# Frame 6: 0.7 -> window=[0.8, 0.1, 0.2, 0.6, 0.7]
#   Base: 1, Hist: 3 -> 1, MA(2.4/5 = 0.48): 0, Med(0.6): 1
# Frame 7: 0.8 -> window=[0.1, 0.2, 0.6, 0.7, 0.8]
#   Base: 1, Hist: 3 -> 1, MA(2.4/5 = 0.48): 0, Med(0.6): 1
# Frame 8: 0.2 -> window=[0.2, 0.6, 0.7, 0.8, 0.2]
#   Base: 0, Hist: 3 -> 1, MA(2.5/5 = 0.50): 1, Med(0.6): 1
# Frame 9: 0.1 -> window=[0.6, 0.7, 0.8, 0.2, 0.1]
#   Base: 0, Hist: 3 -> 1, MA(2.4/5 = 0.48): 0, Med(0.6): 1

CONFIDENCE_10 = [0.1, 0.2, 0.8, 0.1, 0.2, 0.6, 0.7, 0.8, 0.2, 0.1]
TAU = 0.5

EXPECTED_BASELINE = [0, 0, 1, 0, 0, 1, 1, 1, 0, 0]
EXPECTED_HISTORY = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
EXPECTED_MOVING_AVG = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
EXPECTED_MEDIAN = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_baseline_postprocessor() -> None:
    processor = BaselinePostProcessor()
    assert isinstance(processor, PostProcessor)
    decisions = processor.decide(CONFIDENCE_10, TAU)
    assert decisions == EXPECTED_BASELINE


def test_history_tracking_postprocessor() -> None:
    processor = HistoryTrackingPostProcessor(window_size=5, required_hits=3)
    assert isinstance(processor, PostProcessor)
    decisions = processor.decide(CONFIDENCE_10, TAU)
    assert decisions == EXPECTED_HISTORY


def test_moving_average_postprocessor() -> None:
    processor = MovingAveragePostProcessor(window_size=5)
    assert isinstance(processor, PostProcessor)
    decisions = processor.decide(CONFIDENCE_10, TAU)
    assert decisions == EXPECTED_MOVING_AVG


def test_median_filter_postprocessor() -> None:
    processor = MedianFilterPostProcessor(window_size=5)
    assert isinstance(processor, PostProcessor)
    decisions = processor.decide(CONFIDENCE_10, TAU)
    assert decisions == EXPECTED_MEDIAN


def test_single_frame_spike_rejection() -> None:
    # Single frame spike at index 2 (confidence = 0.8)
    # Must be accepted by baseline but rejected by median filter
    baseline_decisions = BaselinePostProcessor().decide(CONFIDENCE_10, TAU)
    median_decisions = MedianFilterPostProcessor().decide(CONFIDENCE_10, TAU)

    spike_idx = 2
    assert baseline_decisions[spike_idx] == 1, "Baseline must accept single-frame spike"
    assert median_decisions[spike_idx] == 0, "Median filter must reject single-frame spike"
