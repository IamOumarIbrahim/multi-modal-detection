#   MMSAR - tests for fusion and the classical filters
#        Oumar Ibrahim U22200741

import pytest

from mmsar.filters import BaselineFilter, HistoryConsensusFilter, MedianFilter, MovingAverageFilter
from mmsar.fusion import late_fusion

# 10 frames with a single frame spike at index 2, tau = 0.5
# worked out by hand, e.g. frame 8: window [0.2 0.6 0.7 0.8 0.2] mean 0.50 median 0.6, 3 hits
CONF = [0.1, 0.2, 0.8, 0.1, 0.2, 0.6, 0.7, 0.8, 0.2, 0.1]
TAU = 0.5


def test_late_fusion():
    assert late_fusion([0.1, 0.9, 0.5], [0.4, 0.2, 0.5]) == [0.4, 0.9, 0.5]
    assert late_fusion([], []) == []
    with pytest.raises(ValueError):
        late_fusion([0.1], [0.1, 0.2])


def test_baseline():
    assert BaselineFilter().decide(CONF, TAU) == [0, 0, 1, 0, 0, 1, 1, 1, 0, 0]


def test_moving_average():
    assert MovingAverageFilter(5).decide(CONF, TAU) == [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]


def test_median():
    # the spike at frame 2 is rejected, the real run from frame 5 is kept
    assert MedianFilter(5).decide(CONF, TAU) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]


def test_median_even_window():
    # 2 frames seen so far -> median is the avrage of the two
    assert MedianFilter(5).decide([0.4, 0.6], 0.5) == [0, 1]


def test_history_consensus():
    assert HistoryConsensusFilter(5, hits=3).decide(CONF, TAU) == [0, 0, 0, 0, 0, 0, 1, 1, 1, 1]
