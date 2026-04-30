"""Q1 dataset assembly + tree baseline + NN trainer on synthetic parquet data."""

from __future__ import annotations

import numpy as np
import torch

from beyond_the_ball.models.datasets import (
    Q1_FLAT_FEATURES,
    Q1_SPATIAL_FEATURES,
    class_weights_from_labels,
    load_q1_table,
    prepare_q1_split,
)
from beyond_the_ball.models.q1_nn import train_q1_nn
from beyond_the_ball.models.q1_tree import train_q1_tree


def test_load_q1_table_joins_labels_flat_and_spatial(synthetic_q1):
    processed, _ = synthetic_q1
    df = load_q1_table(processed, include_spatial=True)
    assert {"match_id", "possession", "label"}.issubset(df.columns)
    for c in Q1_FLAT_FEATURES:
        assert c in df.columns
    for c in Q1_SPATIAL_FEATURES:
        assert c in df.columns


def test_prepare_q1_split_imputes_and_scales(synthetic_q1):
    processed, splits = synthetic_q1
    df = load_q1_table(processed, include_spatial=True)
    data = prepare_q1_split(df, splits, feature_set="flat_spatial", scale=True)

    assert not np.any(np.isnan(data.X_train))
    assert not np.any(np.isnan(data.X_val))
    assert not np.any(np.isnan(data.X_test))
    # StandardScaler -> train means roughly 0, train stds roughly 1.
    np.testing.assert_allclose(data.X_train.mean(axis=0), 0.0, atol=1e-6)
    np.testing.assert_allclose(data.X_train.std(axis=0), 1.0, atol=1e-6)
    assert data.X_train.shape[1] == len(Q1_FLAT_FEATURES) + len(Q1_SPATIAL_FEATURES)


def test_prepare_q1_split_flat_only_skips_spatial(synthetic_q1):
    processed, splits = synthetic_q1
    df = load_q1_table(processed, include_spatial=False)
    data = prepare_q1_split(df, splits, feature_set="flat", scale=False)
    assert data.X_train.shape[1] == len(Q1_FLAT_FEATURES)
    assert data.feature_names == Q1_FLAT_FEATURES


def test_prepare_q1_split_no_match_leakage(synthetic_q1):
    processed, splits = synthetic_q1
    df = load_q1_table(processed, include_spatial=False)
    data = prepare_q1_split(df, splits, feature_set="flat", scale=False)

    train_matches = set(data.keys_train["match_id"].astype(int).tolist())
    val_matches = set(data.keys_val["match_id"].astype(int).tolist())
    test_matches = set(data.keys_test["match_id"].astype(int).tolist())
    assert train_matches.isdisjoint(val_matches)
    assert train_matches.isdisjoint(test_matches)
    assert val_matches.isdisjoint(test_matches)
    assert train_matches == set(splits.train)


def test_class_weights_match_sklearn_balanced_heuristic():
    y = np.array([0, 0, 0, 1, 1, 2])
    w = class_weights_from_labels(y, n_classes=3)
    expected = np.array([6 / (3 * 3), 6 / (3 * 2), 6 / (3 * 1)])
    np.testing.assert_allclose(w, expected)


def test_train_q1_tree_recovers_synthetic_signal(synthetic_q1):
    processed, splits = synthetic_q1
    result = train_q1_tree(
        splits,
        processed_dir=processed,
        max_depths=(3, 5, 10, None),
        seed=0,
    )
    assert result.best_max_depth in {3, 5, 10, None}
    assert len(result.sweep) == 4
    # Synthetic rule is fully separable -> tree should ace val + test.
    assert result.val_macro_f1 >= 0.95
    assert float(result.test_metrics["macro_f1"]) >= 0.95


def test_train_q1_nn_runs_and_beats_random(synthetic_q1):
    processed, splits = synthetic_q1
    result = train_q1_nn(
        splits,
        processed_dir=processed,
        epochs=40,
        patience=10,
        batch_size=64,
        seed=0,
        device=torch.device("cpu"),
    )
    assert result.val_proba.shape[1] == 3
    assert result.test_proba.shape[1] == 3
    np.testing.assert_allclose(result.val_proba.sum(axis=1), 1.0, atol=1e-5)
    # Random would give macro-F1 ~0.33; learnable signal should land well above.
    assert float(result.val_metrics["macro_f1"]) > 0.5
    assert float(result.test_metrics["macro_f1"]) > 0.5
    assert result.history.best_epoch >= 1
