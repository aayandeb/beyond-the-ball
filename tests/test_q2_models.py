"""Q2 dataset assembly + logreg baseline + NN trainer on synthetic parquet data."""

from __future__ import annotations

import numpy as np
import torch

from beyond_the_ball.models.datasets import (
    Q2_FLAT_FEATURES,
    Q2_SPATIAL_FEATURES,
    Q2_ZONE_FEATURES,
    load_q2_table,
    pos_weight_from_labels,
    prepare_q2_split,
)
from beyond_the_ball.models.q2_logreg import train_q2_logreg
from beyond_the_ball.models.q2_nn import train_q2_nn


def test_load_q2_table_joins_labels_flat_and_spatial(synthetic_q2):
    processed, _ = synthetic_q2
    df = load_q2_table(processed, include_spatial=True)
    assert {"match_id", "event_uuid", "label"}.issubset(df.columns)
    for c in Q2_FLAT_FEATURES:
        assert c in df.columns, f"missing flat feature: {c}"
    for c in Q2_SPATIAL_FEATURES:
        assert c in df.columns, f"missing spatial feature: {c}"
    # Zone one-hots sum to 1 per row.
    zone_sum = df[list(Q2_ZONE_FEATURES)].sum(axis=1)
    assert (zone_sum == 1).all()


def test_prepare_q2_split_imputes_scales_and_no_match_leakage(synthetic_q2):
    processed, splits = synthetic_q2
    df = load_q2_table(processed, include_spatial=True)
    data = prepare_q2_split(df, splits, feature_set="flat_spatial", scale=True)

    assert not np.any(np.isnan(data.X_train))
    assert not np.any(np.isnan(data.X_val))
    assert not np.any(np.isnan(data.X_test))
    np.testing.assert_allclose(data.X_train.mean(axis=0), 0.0, atol=1e-6)
    # Non-constant columns (e.g. continuous + most one-hots) should be unit-std after scaling.
    # A zone one-hot with zero variance in train (rare class) stays at std=0; that's fine.
    train_std = data.X_train.std(axis=0)
    nonconstant = train_std > 1e-9
    assert nonconstant.sum() >= len(Q2_FLAT_FEATURES) + len(Q2_SPATIAL_FEATURES) - len(Q2_ZONE_FEATURES)
    np.testing.assert_allclose(train_std[nonconstant], 1.0, atol=1e-6)
    assert data.X_train.shape[1] == len(Q2_FLAT_FEATURES) + len(Q2_SPATIAL_FEATURES)

    train_matches = set(data.keys_train["match_id"].astype(int).tolist())
    val_matches = set(data.keys_val["match_id"].astype(int).tolist())
    test_matches = set(data.keys_test["match_id"].astype(int).tolist())
    assert train_matches.isdisjoint(val_matches)
    assert train_matches.isdisjoint(test_matches)
    assert val_matches.isdisjoint(test_matches)


def test_pos_weight_handles_imbalance_and_edges():
    y = np.array([0, 0, 0, 0, 1])  # 4 neg, 1 pos -> weight 4.0
    assert pos_weight_from_labels(y) == 4.0
    # All positives or all negatives -> 1.0 (avoid blow-up).
    assert pos_weight_from_labels(np.ones(5)) == 1.0
    assert pos_weight_from_labels(np.zeros(5)) == 1.0


def test_train_q2_logreg_beats_base_rate(synthetic_q2):
    processed, splits = synthetic_q2
    result = train_q2_logreg(splits, processed_dir=processed, seed=0)

    base_rate = float(result.test_metrics["positive_rate"])
    test_pr = float(result.test_metrics["pr_auc"])
    test_roc = float(result.test_metrics["roc_auc"])

    # PR-AUC of a random scorer ≈ base rate; ours should be much higher.
    assert test_pr > base_rate + 0.1
    # Strong, learnable signal -> ROC-AUC well above 0.5.
    assert test_roc > 0.7
    # Output probabilities sum to 1 across both columns.
    assert result.val_proba.shape == (len(result.data.y_val),)
    assert (result.val_proba >= 0).all() and (result.val_proba <= 1).all()


def test_train_q2_nn_beats_base_rate(synthetic_q2):
    processed, splits = synthetic_q2
    result = train_q2_nn(
        splits,
        processed_dir=processed,
        epochs=30,
        patience=8,
        batch_size=256,
        seed=0,
        device=torch.device("cpu"),
    )

    base_rate = float(result.test_metrics["positive_rate"])
    test_pr = float(result.test_metrics["pr_auc"])
    test_roc = float(result.test_metrics["roc_auc"])

    assert test_pr > base_rate + 0.05
    assert test_roc > 0.65
    assert result.history.best_epoch >= 1
    assert result.val_proba.shape == (len(result.data.y_val),)
