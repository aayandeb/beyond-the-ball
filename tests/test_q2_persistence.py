"""Save/load roundtrip + metrics CSV append for Q2 logreg and Q2 NN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from beyond_the_ball.models.nn_common import predict_proba
from beyond_the_ball.models.persistence import read_json
from beyond_the_ball.models.q2_logreg import (
    load_q2_logreg,
    save_q2_logreg,
    train_q2_logreg,
)
from beyond_the_ball.models.q2_nn import load_q2_nn, save_q2_nn, train_q2_nn


def _positive_proba_from_loaded(model, X):
    proba = model.predict_proba(X)
    pos_index = int(np.where(model.classes_ == 1)[0][0])
    return proba[:, pos_index]


def test_q2_logreg_save_load_roundtrip(synthetic_q2_dirs):
    processed, splits, artifacts, metrics_csv = synthetic_q2_dirs
    result = train_q2_logreg(splits, processed_dir=processed, seed=0)

    save_q2_logreg(result, base_dir=artifacts, metrics_csv=metrics_csv, split_seed=2)
    loaded = load_q2_logreg(base_dir=artifacts)

    proba_orig = _positive_proba_from_loaded(result.model, result.data.X_test)
    proba_loaded = _positive_proba_from_loaded(loaded.model, result.data.X_test)
    np.testing.assert_allclose(proba_orig, proba_loaded, atol=1e-12)

    manifest = read_json(Path(artifacts) / "q2_logreg" / "manifest.json")
    assert manifest["task"] == "q2"
    assert manifest["feature_set"] == "flat"
    assert manifest["feature_names"] == list(result.data.feature_names)
    assert manifest["C"] == result.C

    df = pd.read_csv(metrics_csv)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["model"] == "q2_logreg"
    assert row["task"] == "q2"
    assert row["test_pr_auc"] == pytest.approx(float(result.test_metrics["pr_auc"]))


def test_q2_nn_save_load_roundtrip(synthetic_q2_dirs):
    processed, splits, artifacts, metrics_csv = synthetic_q2_dirs
    result = train_q2_nn(
        splits,
        processed_dir=processed,
        epochs=20,
        patience=5,
        batch_size=256,
        seed=0,
        device=torch.device("cpu"),
    )

    save_q2_nn(result, base_dir=artifacts, metrics_csv=metrics_csv, split_seed=2)
    loaded = load_q2_nn(base_dir=artifacts, map_location="cpu")

    proba_orig = predict_proba(result.model, result.data.X_test, device=torch.device("cpu"), binary=True)
    proba_loaded = predict_proba(loaded.model, result.data.X_test, device=torch.device("cpu"), binary=True)
    np.testing.assert_allclose(proba_orig, proba_loaded, atol=1e-6)

    manifest = read_json(Path(artifacts) / "q2_nn" / "manifest.json")
    assert manifest["feature_set"] == "flat_spatial"
    assert manifest["architecture"]["out_dim"] == 1
    assert manifest["pos_weight"] == result.pos_weight

    history = read_json(Path(artifacts) / "q2_nn" / "history.json")
    assert "train_loss" in history and "val_metric" in history

    df = pd.read_csv(metrics_csv)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["model"] == "q2_nn"
    extra = json.loads(row["extra"])
    assert extra["best_epoch"] == result.history.best_epoch
    assert extra["pos_weight"] == result.pos_weight
