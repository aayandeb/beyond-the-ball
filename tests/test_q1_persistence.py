"""Save/load roundtrip + metrics CSV append for Q1 tree and Q1 NN."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from beyond_the_ball.models.nn_common import predict_proba
from beyond_the_ball.models.persistence import (
    METRICS_LOG_COLUMNS,
    append_metrics_log,
    read_json,
)
from beyond_the_ball.models.q1_nn import load_q1_nn, save_q1_nn, train_q1_nn
from beyond_the_ball.models.q1_tree import load_q1_tree, save_q1_tree, train_q1_tree


def test_metrics_log_appends_with_correct_header(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    append_metrics_log({"model": "a", "task": "q1", "test_macro_f1": 0.5}, path=csv_path)
    append_metrics_log({"model": "b", "task": "q2", "test_f1": 0.6}, path=csv_path)
    df = pd.read_csv(csv_path)
    assert list(df.columns) == list(METRICS_LOG_COLUMNS)
    assert len(df) == 2
    assert df.iloc[0]["model"] == "a"
    assert df.iloc[1]["model"] == "b"


def test_q1_tree_save_load_roundtrip_predicts_identically(synthetic_q1_dirs):
    processed, splits, artifacts, metrics_csv = synthetic_q1_dirs
    result = train_q1_tree(splits, processed_dir=processed, seed=0)

    save_q1_tree(result, base_dir=artifacts, metrics_csv=metrics_csv, split_seed=1)
    loaded = load_q1_tree(base_dir=artifacts)

    # Same predictions on test set (preprocessor + model both deterministic).
    pred_orig = result.model.predict(result.data.X_test)
    pred_loaded = loaded.model.predict(result.data.X_test)
    np.testing.assert_array_equal(pred_orig, pred_loaded)

    # Manifest captures feature names and chosen depth.
    manifest = read_json(Path(artifacts) / "q1_tree" / "manifest.json")
    assert manifest["feature_set"] == "flat"
    assert manifest["best_max_depth"] == result.best_max_depth
    assert manifest["feature_names"] == list(result.data.feature_names)

    # Metrics CSV got a row.
    df = pd.read_csv(metrics_csv)
    assert len(df) == 1
    assert df.iloc[0]["model"] == "q1_tree"
    assert df.iloc[0]["task"] == "q1"
    assert df.iloc[0]["feature_set"] == "flat"
    assert df.iloc[0]["test_macro_f1"] == pytest.approx(float(result.test_metrics["macro_f1"]))


def test_q1_nn_save_load_roundtrip_predicts_identically(synthetic_q1_dirs):
    processed, splits, artifacts, metrics_csv = synthetic_q1_dirs
    result = train_q1_nn(
        splits,
        processed_dir=processed,
        epochs=20,
        patience=5,
        batch_size=64,
        seed=0,
        device=torch.device("cpu"),
    )

    save_q1_nn(result, base_dir=artifacts, metrics_csv=metrics_csv, split_seed=1)
    loaded = load_q1_nn(base_dir=artifacts, map_location="cpu")

    proba_orig = predict_proba(result.model, result.data.X_test, device=torch.device("cpu"))
    proba_loaded = predict_proba(loaded.model, result.data.X_test, device=torch.device("cpu"))
    np.testing.assert_allclose(proba_orig, proba_loaded, atol=1e-6)

    manifest = read_json(Path(artifacts) / "q1_nn" / "manifest.json")
    assert manifest["feature_set"] == "flat_spatial"
    assert manifest["architecture"]["in_dim"] == result.data.X_train.shape[1]
    assert manifest["architecture"]["out_dim"] == len(result.data.classes)
    assert manifest["architecture"]["hidden"] == [64, 32]

    history = read_json(Path(artifacts) / "q1_nn" / "history.json")
    assert "train_loss" in history and "val_loss" in history and "val_metric" in history
    assert history["best_epoch"] >= 1

    df = pd.read_csv(metrics_csv)
    assert len(df) == 1
    assert df.iloc[0]["model"] == "q1_nn"
    extra = json.loads(df.iloc[0]["extra"])
    assert extra["best_epoch"] == result.history.best_epoch
