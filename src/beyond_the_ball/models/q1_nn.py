"""Q1 NN: MLP on per-possession flat + spatial features.

Input: scaled, median-imputed (flat + spatial-mean3) per-possession features.
Architecture: ``[in -> 64 -> 32 -> 3]`` MLP with ReLU and dropout.
Loss: cross-entropy with class weights (matching ``class_weight='balanced'``).
Optimizer: Adam. Early stopping on val macro-F1.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from sklearn.pipeline import Pipeline
from torch import nn

from ..data.splits import MatchSplits
from ..eval.metrics import q1_metrics
from .datasets import (
    DEFAULT_PROCESSED_DIR,
    Q1SplitData,
    class_weights_from_labels,
    ensure_feature_tables,
    load_q1_table,
    prepare_q1_split,
)
from .nn_common import MLP, TrainHistory, pick_device, predict_proba, train_classifier
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
class Q1NNResult:
    """Output of a Q1 NN training run."""

    model: nn.Module
    data: Q1SplitData
    history: TrainHistory
    val_metrics: dict[str, object]
    test_metrics: dict[str, object]
    val_proba: np.ndarray
    test_proba: np.ndarray
    architecture: dict[str, object]


def _idx_to_label(classes: Sequence[str], y: np.ndarray) -> np.ndarray:
    classes = list(classes)
    return np.array([classes[i] for i in y], dtype=object)


def _val_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Operates on integer class indices; q1_metrics is label-agnostic.
    return float(q1_metrics(y_true, y_pred, labels=sorted(np.unique(y_true).tolist()))["macro_f1"])


def train_q1_nn(
    splits: MatchSplits,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    canonical_path: str | Path | None = None,
    hidden: Sequence[int] = (64, 32),
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 256,
    epochs: int = 200,
    patience: int = 15,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Q1NNResult:
    """Train the Q1 NN on flat + spatial possession features."""
    if canonical_path is not None:
        ensure_feature_tables(canonical_path=canonical_path, processed_dir=processed_dir)

    df = load_q1_table(processed_dir, include_spatial=True)
    data = prepare_q1_split(df, splits, feature_set="flat_spatial", scale=True)

    n_classes = len(data.classes)
    class_w = class_weights_from_labels(data.y_train, n_classes=n_classes)

    device = device or pick_device()
    weight_tensor = torch.as_tensor(class_w, dtype=torch.float32, device=device)
    loss_fn = nn.CrossEntropyLoss(weight=weight_tensor)

    architecture: dict[str, object] = {
        "type": "MLP",
        "in_dim": int(data.X_train.shape[1]),
        "out_dim": int(n_classes),
        "hidden": list(hidden),
        "dropout": float(dropout),
    }
    model = MLP(in_dim=data.X_train.shape[1], out_dim=n_classes, hidden=hidden, dropout=dropout)

    model, history = train_classifier(
        model,
        data.X_train,
        data.y_train,
        data.X_val,
        data.y_val,
        loss_fn=loss_fn,
        val_metric_fn=_val_macro_f1,
        target_dtype=torch.long,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        seed=seed,
        verbose=verbose,
    )

    val_proba = predict_proba(model, data.X_val, device=device)
    test_proba = predict_proba(model, data.X_test, device=device)
    val_pred_idx = val_proba.argmax(axis=1)
    test_pred_idx = test_proba.argmax(axis=1)

    val_true = _idx_to_label(data.classes, data.y_val)
    test_true = _idx_to_label(data.classes, data.y_test)
    val_pred = _idx_to_label(data.classes, val_pred_idx)
    test_pred = _idx_to_label(data.classes, test_pred_idx)

    return Q1NNResult(
        model=model,
        data=data,
        history=history,
        val_metrics=q1_metrics(val_true, val_pred, labels=data.classes),
        test_metrics=q1_metrics(test_true, test_pred, labels=data.classes),
        val_proba=val_proba,
        test_proba=test_proba,
        architecture=architecture,
    )


@dataclass
class LoadedQ1NN:
    """Materialized Q1 NN artifact loaded from disk."""

    model: nn.Module
    preprocessor: Pipeline
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    val_metrics: dict[str, object]
    test_metrics: dict[str, object]
    history: dict[str, object]
    architecture: dict[str, object]


def save_q1_nn(
    result: Q1NNResult,
    *,
    name: str = "q1_nn",
    base_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    split_seed: int | None = None,
    architecture: dict[str, object] | None = None,
) -> Path:
    """Persist state_dict + preprocessor + metrics + history; append metrics CSV."""
    out_dir = model_dir(name, base=base_dir)
    torch.save(result.model.state_dict(), out_dir / "model.pt")
    joblib.dump(result.data.preprocessor, out_dir / "preprocessor.joblib")

    arch = architecture or result.architecture

    write_json(out_dir / "metrics.json", {
        "val": result.val_metrics,
        "test": result.test_metrics,
    })
    write_json(out_dir / "history.json", asdict(result.history))
    write_json(out_dir / "manifest.json", {
        "model": name,
        "task": "q1",
        "feature_set": "flat_spatial",
        "feature_names": list(result.data.feature_names),
        "classes": list(result.data.classes),
        "architecture": arch,
        "saved_at": utc_timestamp(),
    })

    append_metrics_log({
        "timestamp": utc_timestamp(),
        "model": name,
        "task": "q1",
        "feature_set": "flat_spatial",
        "split_seed": split_seed,
        "n_train": int(len(result.data.y_train)),
        "n_val": int(len(result.data.y_val)),
        "n_test": int(len(result.data.y_test)),
        "val_metric_name": "macro_f1",
        "val_metric": result.history.best_metric,
        "test_accuracy": float(result.test_metrics["accuracy"]),
        "test_macro_f1": float(result.test_metrics["macro_f1"]),
        "extra": {
            "best_epoch": result.history.best_epoch,
            "n_epochs_run": len(result.history.train_loss),
            "architecture": arch,
        },
    }, path=metrics_csv)

    return out_dir


def _build_mlp_from_arch(arch: dict[str, object]) -> nn.Module:
    return MLP(
        in_dim=int(arch["in_dim"]),
        out_dim=int(arch["out_dim"]),
        hidden=tuple(arch.get("hidden", (64, 32))),
        dropout=float(arch.get("dropout", 0.2)),
    )


def load_q1_nn(
    *,
    name: str = "q1_nn",
    base_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    map_location: str | torch.device = "cpu",
) -> LoadedQ1NN:
    """Load a previously saved Q1 NN artifact (weights into a fresh MLP)."""
    src = Path(base_dir) / name
    manifest = read_json(src / "manifest.json")
    metrics = read_json(src / "metrics.json")
    history = read_json(src / "history.json")
    arch = manifest["architecture"]

    model = _build_mlp_from_arch(arch)
    state = torch.load(src / "model.pt", map_location=map_location, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    preprocessor = joblib.load(src / "preprocessor.joblib")

    return LoadedQ1NN(
        model=model,
        preprocessor=preprocessor,
        feature_names=tuple(manifest["feature_names"]),
        classes=tuple(manifest["classes"]),
        val_metrics=metrics["val"],
        test_metrics=metrics["test"],
        history=history,
        architecture=arch,
    )
