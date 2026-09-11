"""Metrics and evaluation package."""

from mmsar.metrics.frame_metrics import calculate_frame_metrics, count_false_alarms
from mmsar.metrics.alarm_metrics import calculate_alarm_metrics, count_alarm_false_triggers
from mmsar.metrics.report import (
    to_markdown_table,
    TABLE_1A_COLUMNS,
    TABLE_1B_COLUMNS,
    TABLE_2_COLUMNS,
    TABLE_3_COLUMNS,
    RESULTS_TABLE_COLUMNS,
)

__all__ = [
    "calculate_frame_metrics",
    "count_false_alarms",
    "calculate_alarm_metrics",
    "count_alarm_false_triggers",
    "to_markdown_table",
    "TABLE_1A_COLUMNS",
    "TABLE_1B_COLUMNS",
    "TABLE_2_COLUMNS",
    "TABLE_3_COLUMNS",
    "RESULTS_TABLE_COLUMNS",
]
