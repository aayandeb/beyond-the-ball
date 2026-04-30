"""Metric helpers for Q1 (multiclass) and Q2 (binary), plus bootstrap CIs."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

Q1_LABELS: tuple[str, ...] = ("shot", "final_third", "turnover")


def q1_metrics(
    y_true: Sequence[str] | np.ndarray,
    y_pred: Sequence[str] | np.ndarray,
    *,
    labels: Sequence[str] = Q1_LABELS,
) -> dict[str, object]:
    """Accuracy, macro-F1, per-class precision/recall/F1, confusion matrix."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    labels = list(labels)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "per_class": {
            label: {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i, label in enumerate(labels)
        },
        "confusion_matrix": cm.tolist(),
        "labels": labels,
    }


def calibration_bins(
    y_true: np.ndarray,
    y_score: np.ndarray,
    *,
    n_bins: int = 10,
) -> dict[str, list[float]]:
    """Equal-width reliability bins on [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.searchsorted(edges, y_score, side="right") - 1, 0, n_bins - 1)
    mean_pred: list[float] = []
    frac_pos: list[float] = []
    counts: list[int] = []
    for b in range(n_bins):
        mask = idx == b
        if not np.any(mask):
            mean_pred.append(float("nan"))
            frac_pos.append(float("nan"))
            counts.append(0)
            continue
        mean_pred.append(float(y_score[mask].mean()))
        frac_pos.append(float(y_true[mask].mean()))
        counts.append(int(mask.sum()))
    return {"mean_predicted": mean_pred, "fraction_positive": frac_pos, "count": counts}


def q2_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_score: Sequence[float] | np.ndarray,
    *,
    threshold: float = 0.5,
    n_calibration_bins: int = 10,
) -> dict[str, object]:
    """Accuracy, F1 at threshold, ROC-AUC, PR-AUC, Brier, calibration bins.

    ``y_score`` must be probabilities (not logits) in [0, 1].
    """
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)
    y_pred = (y_score >= threshold).astype(int)

    only_one_class = len(np.unique(y_true)) < 2
    roc = float("nan") if only_one_class else float(roc_auc_score(y_true, y_score))
    pr = float("nan") if only_one_class else float(average_precision_score(y_true, y_score))

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": roc,
        "pr_auc": pr,
        "brier": float(brier_score_loss(y_true, y_score)),
        "positive_rate": float(y_true.mean()),
        "threshold": float(threshold),
        "calibration": calibration_bins(y_true, y_score, n_bins=n_calibration_bins),
    }


def bootstrap_ci(
    metric_fn: Callable[..., float],
    *arrays: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, float]:
    """Percentile bootstrap CI for a scalar metric over paired arrays.

    Each resample draws indices with replacement and applies ``metric_fn`` to
    the resampled arrays in the same order. Returns ``{"mean", "lo", "hi"}``.
    NaN resamples (e.g. AUC when a bootstrap sample has one class) are dropped.
    """
    if not arrays:
        raise ValueError("bootstrap_ci needs at least one array.")
    arrays = tuple(np.asarray(a) for a in arrays)
    n = len(arrays[0])
    if any(len(a) != n for a in arrays):
        raise ValueError("All arrays must share the same length.")
    if n == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}

    rng = np.random.default_rng(seed)
    stats = np.empty(n_resamples, dtype=float)
    for i in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        try:
            stats[i] = float(metric_fn(*(a[idx] for a in arrays)))
        except (ValueError, ZeroDivisionError):
            stats[i] = np.nan
    valid = stats[~np.isnan(stats)]
    if len(valid) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    lo, hi = np.quantile(valid, [alpha / 2, 1 - alpha / 2])
    return {"mean": float(valid.mean()), "lo": float(lo), "hi": float(hi)}
