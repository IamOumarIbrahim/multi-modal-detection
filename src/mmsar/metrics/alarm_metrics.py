"""Alarm-level event detection and evaluation metrics."""

from typing import Sequence, Optional, Any


def _extract_events(binary_seq: Sequence[int]) -> list[tuple[int, int]]:
    """Extract contiguous [start, end) intervals of 1s from a binary sequence."""
    events: list[tuple[int, int]] = []
    in_event = False
    start = 0

    for i, val in enumerate(binary_seq):
        if val == 1 and not in_event:
            in_event = True
            start = i
        elif val == 0 and in_event:
            in_event = False
            events.append((start, i))

    if in_event:
        events.append((start, len(binary_seq)))

    return events


def calculate_alarm_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Calculate event-level alarm precision, recall, and F1-score.

    An alarm event is a contiguous sequence of frames with alarm = 1.
    - True Positive (TP): A predicted alarm event that temporally overlaps with at least
      one ground-truth event.
    - False Positive (FP): A predicted alarm event that does not overlap any ground-truth event.
    - False Negative (FN): A ground-truth event that is not overlapped by any predicted alarm.

    Args:
        y_true: Ground truth binary sequence.
        y_pred: Predicted binary sequence.

    Returns:
        Dict with keys: "precision", "recall", "f1", "tp", "fp", "fn",
        "num_pred_alarms", "num_true_alarms".
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)}"
        )

    true_events = _extract_events(y_true)
    pred_events = _extract_events(y_pred)

    if not pred_events and not true_events:
        return {
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "tp": 0.0,
            "fp": 0.0,
            "fn": 0.0,
            "num_pred_alarms": 0.0,
            "num_true_alarms": 0.0,
        }

    # TP/FP for predicted alarms
    tp = 0
    fp = 0
    for p_start, p_end in pred_events:
        # Check if overlaps with any true event
        overlaps = any(
            max(p_start, t_start) < min(p_end, t_end)
            for t_start, t_end in true_events
        )
        if overlaps:
            tp += 1
        else:
            fp += 1

    # FN for true events
    fn = 0
    for t_start, t_end in true_events:
        detected = any(
            max(p_start, t_start) < min(p_end, t_end)
            for p_start, p_end in pred_events
        )
        if not detected:
            fn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "num_pred_alarms": float(len(pred_events)),
        "num_true_alarms": float(len(true_events)),
    }


def count_alarm_false_triggers(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    """Count false alarm triggers at event level, groupable by an arbitrary label.

    Args:
        y_true: Ground truth binary sequence.
        y_pred: Predicted binary sequence.
        labels: Optional label per frame. The label at the start of each false alarm
                event is used for categorization.

    Returns:
        Dict with total false alarm triggers and grouped counts.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)}"
        )
    if labels is not None and len(labels) != len(y_true):
        raise ValueError(
            f"Length mismatch: labels has {len(labels)} items, y_true has {len(y_true)}"
        )

    true_events = _extract_events(y_true)
    pred_events = _extract_events(y_pred)

    counts: dict[str, int] = {"total": 0}

    for p_start, p_end in pred_events:
        overlaps = any(
            max(p_start, t_start) < min(p_end, t_end)
            for t_start, t_end in true_events
        )
        if not overlaps:
            counts["total"] += 1
            if labels is not None:
                lbl = labels[p_start]
                counts[lbl] = counts.get(lbl, 0) + 1

    return counts
