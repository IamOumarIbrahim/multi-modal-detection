"""Reporting utilities and Markdown table formatting matching README.md results tables."""

from typing import Sequence, Any, Union

TABLE_1_COLUMNS: list[str] = [
    "Configuration / Condition",
    "Modality",
    "Test Environment",
    "Precision",
    "Recall",
    "F1-Score",
]

TABLE_2_COLUMNS: list[str] = [
    "Method",
    r"Val $\tau^*$",
    "Precision",
    "Recall",
    "F1-Score",
    r"F1 (95% CI)$^*$",
    "Pos. Recall (%)",
    "Negative FP",
    "FASR (%)",
    "Latency / Frame",
]

TABLE_3_COLUMNS: list[str] = [
    "Method",
    "Environment",
    "FASR (%)",
    "FP Avoided vs. M1",
    r"Bandwidth Saved (kB)$^\dagger$",
    r"Battery Saved (kJ)$^\ddagger$",
    "Est. Added Flight Time (min)",
]

# Aliases for backward compatibility
TABLE_1A_COLUMNS: list[str] = TABLE_1_COLUMNS
TABLE_1B_COLUMNS: list[str] = TABLE_2_COLUMNS

RESULTS_TABLE_COLUMNS: list[list[str]] = [
    TABLE_1_COLUMNS,
    TABLE_2_COLUMNS,
    TABLE_3_COLUMNS,
]

DOCSTRING_TABLE_CONSTANTS = """
README Results Table Column Sets:
1. Upstream Frame-Level Detection Performance (Table 1):
   ['Configuration / Condition', 'Modality', 'Test Environment', 'Precision', 'Recall', 'F1-Score']
2. Comparative Benchmark of Causal Post-Processing (Table 2):
   ['Method', 'Val tau*', 'Precision', 'Recall', 'F1-Score', 'F1 (95% CI)*', 'Pos. Recall (%)', 'Negative FP', 'FASR (%)', 'Latency / Frame']
3. Environment-Stratified Operational Resource Impact (Table 3):
   ['Method', 'Environment', 'FASR (%)', 'FP Avoided vs. M1', 'Bandwidth Saved (kB)', 'Battery Saved (kJ)', 'Est. Added Flight Time (min)']
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
