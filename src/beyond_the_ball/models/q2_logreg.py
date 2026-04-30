"""Q2 baseline: LogisticRegression on per-event flat features.

L2 penalty + ``class_weight='balanced'`` to compensate for the heavy class
imbalance (~5% positives). The fitted ``StandardScaler`` is part of the
preprocessor pipeline persisted alongside the estimator, so saved artifacts
are self-contained.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..data.splits import MatchSplits
from ..eval.metrics import q2_metrics
from .datasets import (
    DEFAULT_PROCESSED_DIR,
    Q2SplitData,
    ensure_feature_tables,
    load_q2_table,
    prepare_q2_split,
)
from .persistence import (
    DEFAULT_ARTIFACTS_DIR,
    DEFAULT_METRICS_CSV,
    append_metrics_log,
    model_dir,
    read_json,
    utc_timestamp,
    write_json,
)


@dataclass
class Q2LogregResult:
    """Output of a Q2 logistic-regression training run."""

    model: LogisticRegression
    data: Q2SplitData
    val_metrics: dict[str, object]
    test_metrics: dict[str, object]
    val_proba: np.ndarray
    test_proba: np.ndarray
    C: float
    threshold: float


def _positive_proba(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    proba = model.predict_proba(X)
    pos_index = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, pos_index]


def train_q2_logreg(
    splits: MatchSplits,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    canonical_path: str | Path | None = None,
    C: float = 1.0,
    max_iter: int = 1000,
    threshold: float = 0.5,
    seed: int = 42,
) -> Q2LogregResult:
    """Train the Q2 logreg baseline on per-event flat features."""
    if canonical_path is not None:
        ensure_feature_tables(canonical_path=canonical_path, processed_dir=processed_dir)

    df = load_q2_table(processed_dir, include_spatial=False)
    data = prepare_q2_split(df, splits, feature_set="flat", scale=True)

    # ``penalty='l2'`` is the default; pass C explicitly so it stays in the manifest.
    clf = LogisticRegression(
        C=C,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=max_iter,
        random_state=seed,
    )
    clf.fit(data.X_train, data.y_train)

    val_proba = _positive_proba(clf, data.X_val)
    test_proba = _positive_proba(clf, data.X_test)

    return Q2LogregResult(
        model=clf,
        data=data,
        val_metrics=q2_metrics(data.y_val, val_proba, threshold=threshold),
        test_metrics=q2_metrics(data.y_test, test_proba, threshold=threshold),
        val_proba=val_proba,
        test_proba=test_proba,
        C=C,
        threshold=threshold,
    )


@dataclass
class LoadedQ2Logreg:
    """Materialized Q2 logreg artifact loaded from disk."""

    model: LogisticRegression
    preprocessor: Pipeline
    feature_names: tuple[str, ...]
    val_metrics: dict[str, object]
    test_metrics: dict[str, object]
    C: float
    threshold: float


def save_q2_logreg(
    result: Q2LogregResult,
    *,
    name: str = "q2_logreg",
    base_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    split_seed: int | None = None,
) -> Path:
    """Persist model + preprocessor + metrics, append to metrics CSV."""
    out_dir = model_dir(name, base=base_dir)
    joblib.dump(result.model, out_dir / "model.joblib")
    joblib.dump(result.data.preprocessor, out_dir / "preprocessor.joblib")
    write_json(out_dir / "metrics.json", {
        "val": result.val_metrics,
        "test": result.test_metrics,
        "C": result.C,
        "threshold": result.threshold,
    })
    write_json(out_dir / "manifest.json", {
        "model": name,
        "task": "q2",
        "feature_set": "flat",
        "feature_names": list(result.data.feature_names),
        "C": result.C,
        "threshold": result.threshold,
        "saved_at": utc_timestamp(),
    })

    test = result.test_metrics
    append_metrics_log({
        "timestamp": utc_timestamp(),
        "model": name,
        "task": "q2",
        "feature_set": "flat",
        "split_seed": split_seed,
        "n_train": int(len(result.data.y_train)),
        "n_val": int(len(result.data.y_val)),
        "n_test": int(len(result.data.y_test)),
        "val_metric_name": "pr_auc",
        "val_metric": float(result.val_metrics["pr_auc"]),
        "test_accuracy": float(test["accuracy"]),
        "test_f1": float(test["f1"]),
        "test_roc_auc": float(test["roc_auc"]),
        "test_pr_auc": float(test["pr_auc"]),
        "extra": {"C": result.C, "threshold": result.threshold,
                  "positive_rate_test": float(test["positive_rate"])},
    }, path=metrics_csv)

    return out_dir


def load_q2_logreg(
    *,
    name: str = "q2_logreg",
    base_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> LoadedQ2Logreg:
    """Load a previously saved Q2 logreg artifact."""
    src = Path(base_dir) / name
    model = joblib.load(src / "model.joblib")
    preprocessor = joblib.load(src / "preprocessor.joblib")
    metrics = read_json(src / "metrics.json")
    manifest = read_json(src / "manifest.json")
    return LoadedQ2Logreg(
        model=model,
        preprocessor=preprocessor,
        feature_names=tuple(manifest["feature_names"]),
        val_metrics=metrics["val"],
        test_metrics=metrics["test"],
        C=float(manifest["C"]),
        threshold=float(manifest["threshold"]),
    )
