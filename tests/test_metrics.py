import math

import numpy as np
import pytest
from sklearn.metrics import f1_score, roc_auc_score

from beyond_the_ball.eval.metrics import (
    Q1_LABELS,
    bootstrap_ci,
    calibration_bins,
    q1_metrics,
    q2_metrics,
)


def test_q1_metrics_perfect_prediction():
    y_true = ["shot", "final_third", "turnover", "shot"]
    y_pred = ["shot", "final_third", "turnover", "shot"]

    res = q1_metrics(y_true, y_pred)

    assert res["accuracy"] == 1.0
    assert res["macro_f1"] == 1.0
    for label in Q1_LABELS:
        per = res["per_class"][label]
        if per["support"] > 0:
            assert per["precision"] == 1.0
            assert per["recall"] == 1.0


def test_q1_metrics_confusion_matrix_shape_and_labels_order():
    y_true = ["shot", "shot", "final_third", "turnover"]
    y_pred = ["shot", "final_third", "final_third", "shot"]
    res = q1_metrics(y_true, y_pred)

    cm = np.array(res["confusion_matrix"])
    assert cm.shape == (3, 3)
    assert res["labels"] == list(Q1_LABELS)
    # shot row: 1 correct shot, 1 misclassified as final_third, 0 turnover.
    assert cm[0].tolist() == [1, 1, 0]


def test_q2_metrics_perfect_separation():
    y_true = np.array([0, 0, 1, 1])
    y_score = np.array([0.1, 0.2, 0.8, 0.9])
    res = q2_metrics(y_true, y_score)

    assert res["accuracy"] == 1.0
    assert res["f1"] == 1.0
    assert res["roc_auc"] == 1.0
    assert res["pr_auc"] == 1.0
    assert res["positive_rate"] == 0.5
    assert res["brier"] < 0.05


def test_q2_metrics_handles_single_class():
    y_true = np.array([1, 1, 1])
    y_score = np.array([0.7, 0.6, 0.8])
    res = q2_metrics(y_true, y_score)

    assert math.isnan(res["roc_auc"])
    assert math.isnan(res["pr_auc"])
    assert res["accuracy"] == 1.0


def test_calibration_bins_aggregate_correctly():
    y_true = np.array([0, 1, 0, 1, 1, 0])
    y_score = np.array([0.05, 0.15, 0.55, 0.65, 0.95, 0.05])
    bins = calibration_bins(y_true, y_score, n_bins=10)

    assert len(bins["mean_predicted"]) == 10
    assert sum(bins["count"]) == len(y_true)
    # Bin 0 (0.0..0.1): two events with score 0.05, one positive -> frac_pos 0.0.
    assert bins["count"][0] == 2
    assert bins["fraction_positive"][0] == 0.0
    # Bin 9 (0.9..1.0): one event with score 0.95, label 1.
    assert bins["count"][9] == 1
    assert bins["fraction_positive"][9] == 1.0


def test_bootstrap_ci_recovers_known_metric():
    rng = np.random.default_rng(0)
    n = 500
    y_true = rng.integers(0, 2, size=n)
    y_pred = y_true.copy()
    flip = rng.choice(n, size=50, replace=False)
    y_pred[flip] = 1 - y_pred[flip]

    point = f1_score(y_true, y_pred)
    ci = bootstrap_ci(f1_score, y_true, y_pred, n_resamples=400, seed=1)

    assert ci["lo"] <= ci["mean"] <= ci["hi"]
    assert ci["lo"] <= point <= ci["hi"]
    assert ci["hi"] - ci["lo"] < 0.2  # tight enough on n=500


def test_bootstrap_ci_is_deterministic():
    # Noisy enough that resamples yield different AUCs, so seeds matter.
    rng = np.random.default_rng(0)
    n = 200
    y_true = rng.integers(0, 2, size=n)
    y_score = 0.5 + 0.1 * y_true + 0.3 * rng.standard_normal(n)

    a = bootstrap_ci(roc_auc_score, y_true, y_score, n_resamples=200, seed=42)
    b = bootstrap_ci(roc_auc_score, y_true, y_score, n_resamples=200, seed=42)
    c = bootstrap_ci(roc_auc_score, y_true, y_score, n_resamples=200, seed=43)

    assert a == b
    assert a != c


def test_bootstrap_ci_validates_array_lengths():
    with pytest.raises(ValueError):
        bootstrap_ci(f1_score, np.array([0, 1, 1]), np.array([0, 1]))
