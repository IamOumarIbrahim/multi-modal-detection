"""Behavioral unit tests for publication benchmark reporting safeguards and diagnostics."""

from pathlib import Path
import pytest
from scripts.finish_and_report_benchmarks import (
    DuplicateMetricError,
    check_duplicate_metrics_against_prior_reports,
    compute_multiseed_statistics,
    compute_bootstrap_ci,
    compute_empirical_iou_jitter,
    compute_confidence_calibration,
    generate_conditioned_verdict,
    parse_ultralytics_val_result,
    build_publication_report,
)


def test_duplicate_metric_safeguard_flags_historical_matches() -> None:
    """Assert duplicate diff checker catches metrics that match prior reports to 4 decimal places."""
    repo_root = Path(__file__).resolve().parent.parent
    prior_reports = [
        repo_root / "YOLO_11n_DESERT_ALL.md",
        repo_root / "EXPERIMENT_MULTI_SEED_REPORT.md",
    ]
    prior_reports = [p for p in prior_reports if p.exists()]
    if not prior_reports:
        pytest.skip("Prior report markdown files not found")

    # 0.9935 is a known historical number from EXPERIMENT_MULTI_SEED_REPORT.md
    colliding_metrics = {
        "new_arm::test::mAP50": 0.9935,
    }

    # Must raise DuplicateMetricError when duplicate diff safeguard is active
    with pytest.raises(DuplicateMetricError):
        check_duplicate_metrics_against_prior_reports(
            new_metrics=colliding_metrics,
            prior_report_paths=prior_reports,
            tolerance=1e-4,
            raise_on_duplicate=True,
        )

    # When raise_on_duplicate is False, returns flagged duplicate entries
    dups = check_duplicate_metrics_against_prior_reports(
        new_metrics=colliding_metrics,
        prior_report_paths=prior_reports,
        tolerance=1e-4,
        raise_on_duplicate=False,
    )
    assert len(dups) >= 1
    assert dups[0]["new_value"] == 0.9935


def test_duplicate_metric_safeguard_passes_novel_numbers() -> None:
    """Assert distinct metrics pass without raising DuplicateMetricError."""
    repo_root = Path(__file__).resolve().parent.parent
    prior_reports = [
        repo_root / "YOLO_11n_DESERT_ALL.md",
        repo_root / "EXPERIMENT_MULTI_SEED_REPORT.md",
    ]
    prior_reports = [p for p in prior_reports if p.exists()]
    if not prior_reports:
        pytest.skip("Prior report markdown files not found")

    novel_metrics = {
        "arm1::test::mAP50": 0.96317,
        "arm1::test::mAP50-95": 0.52891,
    }

    dups = check_duplicate_metrics_against_prior_reports(
        new_metrics=novel_metrics,
        prior_report_paths=prior_reports,
        tolerance=1e-4,
        raise_on_duplicate=True,
    )
    assert len(dups) == 0


def test_conditioned_verdict_generation_not_templated() -> None:
    """Assert verdict strings are conditioned on actual numbers, never templated."""
    # High recall
    v_high = generate_conditioned_verdict(recall=0.99)
    assert "Resilient" in v_high

    # Moderate recall
    v_mod = generate_conditioned_verdict(recall=0.88)
    assert "Moderate" in v_mod
    assert "88.0%" in v_mod

    # Degraded recall (miss rate)
    v_low = generate_conditioned_verdict(recall=0.75)
    assert "Significant operational degradation" in v_low
    assert "25.0% miss rate" in v_low


def test_parse_ultralytics_val_result_from_live_object() -> None:
    """Assert live Ultralytics result objects are parsed cleanly without placeholder dicts."""
    class MockDetMetrics:
        def __init__(self) -> None:
            self.results_dict = {
                "metrics/precision(B)": 0.985,
                "metrics/recall(B)": 0.942,
                "metrics/mAP50(B)": 0.991,
                "metrics/mAP50-95(B)": 0.524,
            }

    parsed = parse_ultralytics_val_result(MockDetMetrics())
    assert parsed["precision"] == 0.985
    assert parsed["recall"] == 0.942
    assert parsed["mAP50"] == 0.991
    assert parsed["mAP50-95"] == 0.524


def test_multiseed_statistics_computation() -> None:
    """Assert multiseed statistics compute mean, std, and sample size correctly."""
    seed_runs = [
        {"mAP50": 0.980, "mAP50-95": 0.510},
        {"mAP50": 0.990, "mAP50-95": 0.520},
        {"mAP50": 0.985, "mAP50-95": 0.530},
    ]
    stats = compute_multiseed_statistics(seed_runs)

    assert stats["mAP50"]["n"] == 3
    assert stats["mAP50"]["mean"] == 0.985
    assert stats["mAP50"]["std"] == 0.005
    assert stats["mAP50-95"]["mean"] == 0.520


def test_empirical_iou_jitter_analysis() -> None:
    """Assert empirical IoU jitter analysis computes real boundary error curves."""
    # 2 boxes: pred vs gt
    preds = [
        (320.0, 320.0, 50.0, 100.0),
        (200.0, 200.0, 40.0, 80.0),
    ]
    gts = [
        (322.0, 322.0, 48.0, 98.0),  # Minor 2px shift
        (204.0, 204.0, 40.0, 80.0),  # 4px translation
    ]

    jitter = compute_empirical_iou_jitter(preds, gts)
    assert jitter["mean_dx_px"] > 0
    assert jitter["mean_dy_px"] > 0
    assert "empirical_iou_drop" in jitter
    assert "iou_50" in jitter["empirical_iou_drop"]
    assert jitter["mean_empirical_iou"] > 0.70


def test_confidence_calibration_ece() -> None:
    """Assert calibration diagram computes expected calibration error."""
    confidences = [0.95, 0.90, 0.85, 0.80, 0.70, 0.60, 0.50, 0.40]
    matches = [True, True, True, True, True, False, False, False]

    calib = compute_confidence_calibration(confidences, matches, n_bins=5)
    assert "expected_calibration_error" in calib
    assert len(calib["bins"]) == 5
    assert calib["expected_calibration_error"] >= 0.0
