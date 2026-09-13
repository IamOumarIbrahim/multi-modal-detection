"""Tests for frame and alarm metrics, reporting, and README table consistency."""

import re
from pathlib import Path
import pytest
from mmsar.metrics.frame_metrics import calculate_frame_metrics, count_false_alarms
from mmsar.metrics.alarm_metrics import calculate_alarm_metrics, count_alarm_false_triggers
from mmsar.metrics.report import (
    to_markdown_table,
    TABLE_1_COLUMNS,
    TABLE_2_COLUMNS,
    TABLE_3_COLUMNS,
    TABLE_1A_COLUMNS,
    TABLE_1B_COLUMNS,
    RESULTS_TABLE_COLUMNS,
)


def test_frame_metrics_hand_computed() -> None:
    # 10 frames synthetic test:
    # y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    # y_pred = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    # TP: indices 0, 1, 6 -> 3
    # FP: indices 4, 7 -> 2
    # FN: indices 2, 8 -> 2
    # TN: indices 3, 5, 9 -> 3
    # Precision = 3 / (3 + 2) = 0.6
    # Recall    = 3 / (3 + 2) = 0.6
    # F1        = 2 * 0.6 * 0.6 / (0.6 + 0.6) = 0.6
    y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]

    metrics = calculate_frame_metrics(y_true, y_pred)

    assert metrics["tp"] == 3.0
    assert metrics["fp"] == 2.0
    assert metrics["fn"] == 2.0
    assert metrics["tn"] == 3.0
    assert pytest.approx(metrics["precision"]) == 0.6
    assert pytest.approx(metrics["recall"]) == 0.6
    assert pytest.approx(metrics["f1"]) == 0.6


def test_false_alarm_counter_grouped() -> None:
    y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    conditions = [
        "desert", "desert", "desert",
        "forest", "forest", "forest",
        "altitude", "altitude", "altitude", "altitude"
    ]

    counts = count_false_alarms(y_true, y_pred, labels=conditions)
    assert counts["total"] == 2
    assert counts["forest"] == 1
    assert counts["altitude"] == 1
    assert "desert" not in counts or counts["desert"] == 0


def test_to_markdown_table() -> None:
    cols = ["Method", "Precision", "Recall"]
    rows = [
        {"Method": "Baseline", "Precision": "0.85", "Recall": "0.90"},
        {"Method": "Mamba", "Precision": "0.92", "Recall": "0.94"},
    ]
    md = to_markdown_table(rows, cols)
    expected = (
        "| Method | Precision | Recall |\n"
        "| --- | --- | --- |\n"
        "| Baseline | 0.85 | 0.90 |\n"
        "| Mamba | 0.92 | 0.94 |"
    )
    assert md == expected


def test_readme_results_tables_header_equality() -> None:
    """Read the live README.md and assert that extracted Results table headers match report.py constants."""
    readme_path = Path(__file__).resolve().parent.parent / "README.md"
    assert readme_path.exists(), f"README.md not found at {readme_path}"

    content = readme_path.read_text(encoding="utf-8")
    assert "### Results" in content or "## Results" in content, "README.md missing Results section"

    results_section = re.split(r"## Results & Benchmarks|### Results", content)[-1].split("## Quick Reproduction")[0]

    # Regex matching Markdown table headers followed by a separator row (supporting CRLF)
    pattern = r"\|([^\r\n]+)\|\r?\n\|(?:\s*[:-]+[-| :]*)\|"
    matches = re.findall(pattern, results_section)

    assert len(matches) == 3, (
        f"Expected 3 results tables in README.md, found {len(matches)}"
    )

    extracted_headers: list[list[str]] = []
    for m in matches:
        cols = [c.strip() for c in m.split("|")]
        extracted_headers.append(cols)

    # Check each table matches the corresponding constant
    assert extracted_headers[0] == TABLE_1_COLUMNS, (
        f"Table 1 header mismatch:\nExpected: {TABLE_1_COLUMNS}\nFound: {extracted_headers[0]}"
    )
    assert extracted_headers[1] == TABLE_2_COLUMNS, (
        f"Table 2 header mismatch:\nExpected: {TABLE_2_COLUMNS}\nFound: {extracted_headers[1]}"
    )
    assert extracted_headers[2] == TABLE_3_COLUMNS, (
        f"Table 3 header mismatch:\nExpected: {TABLE_3_COLUMNS}\nFound: {extracted_headers[2]}"
    )

    # Verify full list matches
    assert extracted_headers == RESULTS_TABLE_COLUMNS


def test_calculate_time_to_alarm() -> None:
    from mmsar.metrics.alarm_metrics import calculate_time_to_alarm

    y_true = [0, 0, 1, 1, 1, 0, 0, 1, 1, 0]
    # Event 1 starts at 2, predicted at 3 -> delay 1
    # Event 2 starts at 7, predicted at 7 -> delay 0
    y_pred = [0, 0, 0, 1, 1, 0, 0, 1, 0, 0]

    res = calculate_time_to_alarm(y_true, y_pred, fps=24.0)
    assert res["detected_events"] == 2
    assert res["total_true_events"] == 2
    assert res["delays_frames"] == [1, 0]
    assert res["mean_delay_frames"] == 0.5
    assert abs(res["mean_delay_seconds"] - (0.5 / 24.0)) < 1e-4

