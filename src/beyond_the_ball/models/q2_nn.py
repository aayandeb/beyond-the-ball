"""Q2 NN: MLP on per-event flat + spatial features.

Architecture: ``[in -> 64 -> 32 -> 1]`` MLP (ReLU + dropout), sigmoid output.
Loss: ``BCEWithLogitsLoss`` with ``pos_weight = n_neg / n_pos`` from train.
Optimizer: Adam. Early stopping on validation **PR-AUC** (better than ROC-AUC
under the heavy class imbalance).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import torch
from sklearn.metrics import average_precision_score
from sklearn.pipeline import Pipeline
from torch import nn

from ..data.splits import MatchSplits
from ..eval.metrics import q2_metrics
from .datasets import (
    DEFAULT_PROCESSED_DIR,
    Q2SplitData,
    ensure_feature_tables,
    load_q2_table,
    pos_weight_from_labels,
    prepare_q2_split,
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
class Q2NNResult:
    """Output of a Q2 NN training run."""

    model: nn.Module
    data: Q2SplitData
    history: TrainHistory
    val_metrics: dict[str, object]
    test_metrics: dict[str, object]
    val_proba: np.ndarray
    test_proba: np.ndarray
    architecture: dict[str, object]
    pos_weight: float
    threshold: float


def _binary_score_predict(logits: torch.Tensor) -> np.ndarray:
    if logits.ndim == 2 and logits.shape[1] == 1:
        logits = logits.squeeze(1)
    return torch.sigmoid(logits).cpu().numpy()


def _pr_auc_metric(y_true: np.ndarray, y_score: np.ndarray) -> float:
    yt = np.asarray(y_true).ravel()
    ys = np.asarray(y_score).ravel()
    if len(np.unique(yt)) < 2:
        return 0.0
    return float(average_precision_score(yt, ys))


def train_q2_nn(
    splits: MatchSplits,
    *,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    canonical_path: str | Path | None = None,
    hidden: Sequence[int] = (64, 32),
    dropout: float = 0.2,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    batch_size: int = 512,
    epochs: int = 200,
    patience: int = 15,
    threshold: float = 0.5,
    seed: int = 42,
    device: torch.device | None = None,
    verbose: bool = False,
) -> Q2NNResult:
    """Train the Q2 MLP on flat + spatial per-event features."""
    if canonical_path is not None:
        ensure_feature_tables(canonical_path=canonical_path, processed_dir=processed_dir)

    df = load_q2_table(processed_dir, include_spatial=True)
    data = prepare_q2_split(df, splits, feature_set="flat_spatial", scale=True)

    pos_weight = pos_weight_from_labels(data.y_train)
    device = device or pick_device()
    pos_weight_t = torch.tensor([pos_weight], dtype=torch.float32, device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_t)

    architecture: dict[str, object] = {
        "type": "MLP",
        "in_dim": int(data.X_train.shape[1]),
        "out_dim": 1,
        "hidden": list(hidden),
        "dropout": float(dropout),
    }
    model = MLP(in_dim=data.X_train.shape[1], out_dim=1, hidden=hidden, dropout=dropout)

    # BCE expects matching shapes — keep targets as (N, 1) float32 for both train and val.
    y_train_2d = data.y_train.reshape(-1, 1).astype(np.float32)
    y_val_2d = data.y_val.reshape(-1, 1).astype(np.float32)

    model, history = train_classifier(
        model,
        data.X_train,
        y_train_2d,
        data.X_val,
        y_val_2d,
        loss_fn=loss_fn,
        val_metric_fn=_pr_auc_metric,
        val_predict_fn=_binary_score_predict,
        target_dtype=torch.float32,
        epochs=epochs,
        patience=patience,
        lr=lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        device=device,
        seed=seed,
        verbose=verbose,
    )

    val_proba = predict_proba(model, data.X_val, device=device, binary=True)
    test_proba = predict_proba(model, data.X_test, device=device, binary=True)

    return Q2NNResult(
        model=model,
        data=data,
        history=history,
        val_metrics=q2_metrics(data.y_val, val_proba, threshold=threshold),
        test_metrics=q2_metrics(data.y_test, test_proba, threshold=threshold),
        val_proba=val_proba,
        test_proba=test_proba,
        architecture=architecture,
        pos_weight=pos_weight,
        threshold=threshold,
    )


@dataclass
class LoadedQ2NN:
    """Materialized Q2 NN artifact loaded from disk."""

    model: nn.Module
    preprocessor: Pipeline
    feature_names: tuple[str, ...]
    val_metrics: dict[str, object]
    test_metrics: dict[str, object]
    history: dict[str, object]
    architecture: dict[str, object]
    pos_weight: float
    threshold: float


def save_q2_nn(
    result: Q2NNResult,
    *,
    name: str = "q2_nn",
    base_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    metrics_csv: str | Path = DEFAULT_METRICS_CSV,
    split_seed: int | None = None,
) -> Path:
    """Persist state_dict + preprocessor + metrics + history; append metrics CSV."""
    out_dir = model_dir(name, base=base_dir)
    torch.save(result.model.state_dict(), out_dir / "model.pt")
    joblib.dump(result.data.preprocessor, out_dir / "preprocessor.joblib")

    write_json(out_dir / "metrics.json", {
        "val": result.val_metrics,
        "test": result.test_metrics,
        "pos_weight": result.pos_weight,
        "threshold": result.threshold,
    })
    write_json(out_dir / "history.json", asdict(result.history))
    write_json(out_dir / "manifest.json", {
        "model": name,
        "task": "q2",
        "feature_set": "flat_spatial",
        "feature_names": list(result.data.feature_names),
        "architecture": result.architecture,
        "pos_weight": result.pos_weight,
        "threshold": result.threshold,
        "saved_at": utc_timestamp(),
    })

    test = result.test_metrics
    append_metrics_log({
        "timestamp": utc_timestamp(),
        "model": name,
        "task": "q2",
        "feature_set": "flat_spatial",
        "split_seed": split_seed,
        "n_train": int(len(result.data.y_train)),
        "n_val": int(len(result.data.y_val)),
        "n_test": int(len(result.data.y_test)),
        "val_metric_name": "pr_auc",
        "val_metric": result.history.best_metric,
        "test_accuracy": float(test["accuracy"]),
        "test_f1": float(test["f1"]),
        "test_roc_auc": float(test["roc_auc"]),
        "test_pr_auc": float(test["pr_auc"]),
        "extra": {
            "best_epoch": result.history.best_epoch,
            "n_epochs_run": len(result.history.train_loss),
            "architecture": result.architecture,
            "pos_weight": result.pos_weight,
            "positive_rate_test": float(test["positive_rate"]),
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


def load_q2_nn(
    *,
    name: str = "q2_nn",
    base_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
    map_location: str | torch.device = "cpu",
) -> LoadedQ2NN:
    """Load a previously saved Q2 NN artifact (weights into a fresh MLP)."""
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

    return LoadedQ2NN(
        model=model,
        preprocessor=preprocessor,
        feature_names=tuple(manifest["feature_names"]),
        val_metrics=metrics["val"],
        test_metrics=metrics["test"],
        history=history,
        architecture=arch,
        pos_weight=float(manifest["pos_weight"]),
        threshold=float(manifest["threshold"]),
    )
