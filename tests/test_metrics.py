#   MMSAR - tests for the metrics and evaluation helpers
#        Oumar Ibrahim U22200741

from mmsar.evaluation import best_threshold, bootstrap_ci, pooled_f1_ci, score_decisions
from mmsar.metrics import calculate_frame_metrics, count_false_alerts, extract_events, time_to_alarm


def test_frame_metrics():
    # tp = 3 (0, 1, 6)  fp = 2 (4, 7)  fn = 2 (2, 8)  tn = 3
    y_true = [1, 1, 1, 0, 0, 0, 1, 0, 1, 0]
    y_pred = [1, 1, 0, 0, 1, 0, 1, 1, 0, 0]
    m = calculate_frame_metrics(y_true, y_pred)
    assert m["tp"] == 3 and m["fp"] == 2 and m["fn"] == 2 and m["tn"] == 3
    assert abs(m["precision"] - 0.6) < 1e-9
    assert abs(m["recall"] - 0.6) < 1e-9
    assert abs(m["f1"] - 0.6) < 1e-9


def test_events_and_false_alerts():
    assert extract_events([0, 1, 1, 0, 1]) == [(1, 3), (4, 5)]
    gt = [0, 0, 1, 1, 0, 0]
    pred = [1, 0, 1, 1, 1, 1]
    # alert at 0 is false, the run 2..5 is on target for 2,3 and spills over to 4,5 (one false run)
    assert count_false_alerts(gt, pred) == 2
    assert time_to_alarm(gt, [0, 0, 0, 1, 0, 0]) == [1]


def test_best_threshold():
    gt = [[0, 0, 0, 0, 0, 1, 1, 1, 0, 0]]
    conf = [[0.1, 0.15, 0.2, 0.1, 0.15, 0.65, 0.75, 0.80, 0.2, 0.1]]
    tau, f1 = best_threshold(conf, gt)
    # every tau between 0.21 and 0.65 is perfect, the tie goes to the one closest to 0.50
    assert f1 == 1.0
    assert abs(tau - 0.49) < 1e-9 or abs(tau - 0.51) < 1e-9


def test_bootstrap():
    mean, low, high = bootstrap_ci([1.0, 1.0, 1.0])
    assert mean == low == high == 1.0
    low, high = pooled_f1_ci([[1, 0], [0, 1]], [[1, 0], [0, 1]])
    assert low == high == 1.0


def test_score_decisions():
    gt = [[0, 1, 1, 0], [0, 0, 0, 0]]
    dec = [[0, 0, 1, 0], [1, 1, 0, 0]]
    s = score_decisions(dec, gt, ["desert", "snow"])
    assert s["detected"] == 1
    assert s["fa_events"] == 1
    assert s["fp_by_biome"]["snow"] == 2
    assert s["mean_delay_frames"] == 1.0
