"""Reporting utilities and Markdown table formatting matching README.md results tables."""

from typing import Sequence, Any, Union

TABLE_1A_COLUMNS: list[str] = [
    "Model",
    "Modality / Stream",
    "Precision",
    "Recall",
    "F1-Score",
]

TABLE_1B_COLUMNS: list[str] = [
    "Model",
    "Post-Processing Method",
    "Frame Precision",
    "Frame Recall",
    "Alarm Precision",
    "Alarm Recall",
]

TABLE_2_COLUMNS: list[str] = [
    "Post-Processing Method",
    "Clear-Negative False Alarms",
    "Hard-Negative False Alarms",
    "Spurious Alarm Triggers (IoT Impact)",
]

TABLE_3_COLUMNS: list[str] = [
    "Post-Processing Method",
    "Desert False Alarms",
    "Forest False Alarms",
    "Altitude (Snow) False Alarms",
]

RESULTS_TABLE_COLUMNS: list[list[str]] = [
    TABLE_1A_COLUMNS,
    TABLE_1B_COLUMNS,
    TABLE_2_COLUMNS,
    TABLE_3_COLUMNS,
]

DOCSTRING_TABLE_CONSTANTS = """
README Results Table Column Sets:
1. Upstream Detector & Fusion Performance:
   ['Model', 'Modality / Stream', 'Precision', 'Recall', 'F1-Score']
2. Post-Processing Evaluation:
   ['Model', 'Post-Processing Method', 'Frame Precision', 'Frame Recall', 'Alarm Precision', 'Alarm Recall']
3. Negative Scenario Rejection & Downstream Transmission Impact:
   ['Post-Processing Method', 'Clear-Negative False Alarms', 'Hard-Negative False Alarms', 'Spurious Alarm Triggers (IoT Impact)']
4. Environmental Breakdown (False Detections per Condition):
   ['Post-Processing Method', 'Desert False Alarms', 'Forest False Alarms', 'Altitude (Snow) False Alarms']
"""


def to_markdown_table(
    rows: Sequence[Union[dict[str, Any], Sequence[Any]]],
    columns: Sequence[str],
) -> str:
    """Render rows and column names into a GitHub Flavored Markdown table string.

    Args:
        rows: Sequence of rows, where each row is either a dict keyed by column name,
              or a sequence of values in column order.
        columns: Sequence of column header names.

    Returns:
        Formatted markdown table string with header and separator.
    """
    if not columns:
        return ""

    header_line = "| " + " | ".join(str(c) for c in columns) + " |"
    separator_line = "| " + " | ".join("---" for _ in columns) + " |"

    row_lines: list[str] = []
    for r in rows:
        if isinstance(r, dict):
            vals = [str(r.get(c, "")) for c in columns]
        else:
            vals = [str(v) for v in r]
        row_lines.append("| " + " | ".join(vals) + " |")

    lines = [header_line, separator_line] + row_lines
    return "\n".join(lines)
