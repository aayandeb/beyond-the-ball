"""Q1 baseline: DecisionTreeClassifier on per-possession flat features.

Sweeps ``max_depth ∈ {3, 5, 10, None}`` on the validation set and selects the
depth maximizing macro-F1. Uses ``class_weight='balanced'`` to handle the
class imbalance among shot/final_third/turnover.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from ..data.splits import MatchSplits
from ..eval.metrics import q1_metrics
from .datasets import (
    DEFAULT_PROCESSED_DIR,
    Q1SplitData,
    ensure_feature_tables,
    load_q1_table,
    prepare_q1_split,
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

DEFAULT_MAX_DEPTHS: tuple[int | None, ...] = (3, 5, 10, None)


@dataclass
class Q1TreeResult:
    """Output of a Q1 tree training run."""

    model: DecisionTreeClassifier
    data: Q1SplitData
    best_max_depth: int | None
    val_macro_f1: float
    sweep: list[dict[str, float | int | None]]
    val_metrics: dict[str, object]
    test_metrics: dict[str, object]


def _macro_f1_from_q1(metrics: dict[str, object]) -> float:
    return float(metrics["macro_f1"])  # type: ignore[arg-type]


def _idx_to_label(classes: Sequence[str], y: np.ndarray) -> np.ndarray:
    classes = list(classes)
    return np.array([classes[i] for i in y], dtype=object)


def train_q1_tree(
    splits: MatchSplits,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    canonical_path: str | Path | None = None,
    max_depths: Sequence[int | None] = DEFAULT_MAX_DEPTHS,
    seed: int = 42,
) -> Q1TreeResult:
    """Train the Q1 tree baseline. Materializes feature parquets if missing.

    Returns the best estimator (refit on train), val + test metrics, and a
    sweep table of (max_depth, val_macro_f1).
    """
    if canonical_path is not None:
        ensure_feature_tables(canonical_path=canonical_path, processed_dir=processed_dir)

    df = load_q1_table(processed_dir, include_spatial=False)
    data = prepare_q1_split(df, splits, feature_set="flat", scale=False)

    sweep: list[dict[str, float | int | None]] = []
    best_depth: int | None = None
    best_metric = -np.inf
    best_model: DecisionTreeClassifier | None = None

    for depth in max_depths:
        clf = DecisionTreeClassifier(
            max_depth=depth,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(data.X_train, data.y_train)
        val_pred_idx = clf.predict(data.X_val)
        val_pred = _idx_to_label(data.classes, val_pred_idx)
        val_true = _idx_to_label(data.classes, data.y_val)
        m = q1_metrics(val_true, val_pred, labels=data.classes)
        macro = _macro_f1_from_q1(m)
        sweep.append({"max_depth": depth, "val_macro_f1": macro})
        if macro > best_metric:
            best_metric = macro
            best_depth = depth
            best_model = clf

    assert best_model is not None  # max_depths non-empty so at least one iter ran.

    val_pred = _idx_to_label(data.classes, best_model.predict(data.X_val))
    test_pred = _idx_to_label(data.classes, best_model.predict(data.X_test))
    val_true = _idx_to_label(data.classes, data.y_val)
    test_true = _idx_to_label(data.classes, data.y_test)

    return Q1TreeResult(
        model=best_model,
        data=data,
        best_max_depth=best_depth,
        val_macro_f1=best_metric,
        sweep=sweep,
        val_metrics=q1_metrics(val_true, val_pred, labels=data.classes),
        test_metrics=q1_metrics(test_true, test_pred, labels=data.classes),
    )


@dataclass
class LoadedQ1Tree:
    """Materialized Q1 tree artifact loaded from disk."""

    model: DecisionTreeClassifier
    preprocessor: Pipeline
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    best_max_depth: int | None
    val_metrics: dict[str, object]
    test_metrics: dict[str, object]
    sweep: list[dict[str, object]]


def save_q1_tree(
    result: Q1TreeResult,
    *,
    name: str = "q1_tree",
    base_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    split_seed: int | None = None,
) -> Path:
    """Persist model + preprocessor + metrics + sweep, append to metrics CSV."""
    out_dir = model_dir(name, base=base_dir)
    joblib.dump(result.model, out_dir / "model.joblib")
    joblib.dump(result.data.preprocessor, out_dir / "preprocessor.joblib")
    write_json(out_dir / "metrics.json", {
        "val": result.val_metrics,
        "test": result.test_metrics,
        "best_max_depth": result.best_max_depth,
        "val_macro_f1": result.val_macro_f1,
    })
    write_json(out_dir / "sweep.json", result.sweep)
    write_json(out_dir / "manifest.json", {
        "model": name,
        "task": "q1",
        "feature_set": "flat",
        "feature_names": list(result.data.feature_names),
        "classes": list(result.data.classes),
        "best_max_depth": result.best_max_depth,
        "saved_at": utc_timestamp(),
    })

    append_metrics_log({
        "timestamp": utc_timestamp(),
        "model": name,
        "task": "q1",
        "feature_set": "flat",
        "split_seed": split_seed,
        "n_train": int(len(result.data.y_train)),
        "n_val": int(len(result.data.y_val)),
        "n_test": int(len(result.data.y_test)),
        "val_metric_name": "macro_f1",
        "val_metric": result.val_macro_f1,
        "test_accuracy": float(result.test_metrics["accuracy"]),
        "test_macro_f1": float(result.test_metrics["macro_f1"]),
        "extra": {"best_max_depth": result.best_max_depth, "sweep": result.sweep},
    }, path=metrics_csv)

    return out_dir


def load_q1_tree(
    *,
    name: str = "q1_tree",
    base_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> LoadedQ1Tree:
    """Load a previously saved Q1 tree artifact."""
    src = Path(base_dir) / name
    model = joblib.load(src / "model.joblib")
    preprocessor = joblib.load(src / "preprocessor.joblib")
    metrics = read_json(src / "metrics.json")
    sweep = read_json(src / "sweep.json")
    manifest = read_json(src / "manifest.json")
    return LoadedQ1Tree(
        model=model,
        preprocessor=preprocessor,
        feature_names=tuple(manifest["feature_names"]),
        classes=tuple(manifest["classes"]),
        best_max_depth=manifest["best_max_depth"],
        val_metrics=metrics["val"],
        test_metrics=metrics["test"],
        sweep=sweep,
    )
