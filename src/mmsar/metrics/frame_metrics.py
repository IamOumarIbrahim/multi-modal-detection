"""Frame-level detection and evaluation metrics."""

from typing import Sequence, Optional, Any


def calculate_frame_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, float]:
    """Calculate frame-level precision, recall, and F1-score from binary labels.

    Args:
        y_true: Ground truth binary sequence (0 or 1).
        y_pred: Predicted binary sequence (0 or 1).

    Returns:
        Dict with keys: "precision", "recall", "f1", "tp", "fp", "fn", "tn".
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)}"
        )

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)

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
        "tn": float(tn),
    }


def count_false_alarms(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    labels: Optional[Sequence[str]] = None,
) -> dict[str, int]:
    """Count false alarms (where y_true == 0 and y_pred == 1), optionally grouped by labels.

    Args:
        y_true: Ground truth binary sequence.
        y_pred: Predicted binary sequence.
        labels: Optional sequence of metadata strings (e.g. scenario type or condition).

    Returns:
        Dict with total false alarm count and grouped counts by label name.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, y_pred has {len(y_pred)}"
        )
    if labels is not None and len(labels) != len(y_true):
        raise ValueError(
            f"Length mismatch: labels has {len(labels)} items, y_true has {len(y_true)}"
        )

    counts: dict[str, int] = {"total": 0}

    for i, (yt, yp) in enumerate(zip(y_true, y_pred)):
        if yt == 0 and yp == 1:
            counts["total"] += 1
            if labels is not None:
                lbl = labels[i]
                counts[lbl] = counts.get(lbl, 0) + 1

    return counts
