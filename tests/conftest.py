"""Shared pytest fixtures for the test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from beyond_the_ball.data.splits import MatchSplits, split_matches
from beyond_the_ball.features.spatial import SPATIAL_COLUMNS
from beyond_the_ball.models.datasets import Q1_SPATIAL_FEATURES

SYNTHETIC_N_MATCHES = 24
SYNTHETIC_POSS_PER_MATCH = 30
SYNTHETIC_RNG_SEED = 0


def _label_for(max_x: float, x_progression: float) -> str:
    if max_x >= 100.0:
        return "shot"
    if x_progression >= 30.0:
        return "final_third"
    return "turnover"


def make_synthetic_q1_parquets(processed_dir: Path) -> tuple[list[int], pd.DataFrame]:
    """Write synthetic Q1 labels + flat + spatial parquets for tests."""
    rng = np.random.default_rng(SYNTHETIC_RNG_SEED)
    rows = []
    for match_id in range(1, SYNTHETIC_N_MATCHES + 1):
        for poss in range(1, SYNTHETIC_POSS_PER_MATCH + 1):
            mean_x = float(rng.uniform(20, 95))
            max_x = float(rng.uniform(mean_x, 120))
            x_prog = float(rng.uniform(-10, 60))
            n_events = int(rng.integers(3, 15))
            duration_s = float(rng.uniform(2, 60))
            n_passes = int(rng.integers(0, n_events + 1))
            spatial = {
                f"{c}_mean3": float(rng.normal(loc=2.0, scale=1.0))
                for c in SPATIAL_COLUMNS
            }
            label = _label_for(max_x, x_prog)
            rows.append(
                {
                    "match_id": match_id,
                    "possession": poss,
                    "n_events": n_events,
                    "duration_s": duration_s,
                    "x_progression": x_prog,
                    "n_passes": n_passes,
                    "mean_x": mean_x,
                    "max_x": max_x,
                    "label": label,
                    **spatial,
                }
            )
    full = pd.DataFrame(rows)

    labels = full[["match_id", "possession", "n_events", "label"]].copy()
    flat_cols = [
        "match_id", "possession", "n_events", "duration_s",
        "x_progression", "n_passes", "mean_x", "max_x",
    ]
    flat = full[flat_cols].copy()
    spatial_cols = ["match_id", "possession"] + list(Q1_SPATIAL_FEATURES)
    spatial = full[spatial_cols].copy()

    nan_idx = rng.choice(len(spatial), size=len(spatial) // 10, replace=False)
    spatial.loc[nan_idx, list(Q1_SPATIAL_FEATURES)] = np.nan

    processed_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(processed_dir / "labels_q1_possession.parquet", index=False)
    flat.to_parquet(processed_dir / "features_flat_possession.parquet", index=False)
    spatial.to_parquet(processed_dir / "features_spatial_possession.parquet", index=False)

    match_ids = sorted(full["match_id"].unique().tolist())
    return match_ids, full


@pytest.fixture
def synthetic_q1(tmp_path: Path) -> tuple[Path, MatchSplits]:
    """Synthetic Q1 parquet dir + a deterministic match-level split."""
    processed = tmp_path / "processed"
    match_ids, _ = make_synthetic_q1_parquets(processed)
    splits = split_matches(match_ids, val_size=0.2, test_size=0.2, seed=1)
    return processed, splits


@pytest.fixture
def synthetic_q1_dirs(tmp_path: Path) -> tuple[Path, MatchSplits, Path, Path]:
    """Synthetic Q1 parquets + split + artifact dir + metrics CSV path."""
    processed = tmp_path / "processed"
    match_ids, _ = make_synthetic_q1_parquets(processed)
    splits = split_matches(match_ids, val_size=0.2, test_size=0.2, seed=1)
    artifacts = tmp_path / "artifacts"
    metrics_csv = tmp_path / "reports" / "metrics_log.csv"
    return processed, splits, artifacts, metrics_csv


# ---------------------------------------------------------------------------
# Q2 synthetic fixtures
# ---------------------------------------------------------------------------

SYNTHETIC_Q2_N_MATCHES = 16
SYNTHETIC_Q2_EVENTS_PER_MATCH = 600
SYNTHETIC_Q2_RNG_SEED = 1
SYNTHETIC_Q2_ACTION_TYPES = ("Pass", "Carry", "Dribble", "Shot")


def _zone_for(x: float, y: float) -> str:
    band = "def" if x < 40 else ("mid" if x < 80 else "att")
    half = "bottom" if y < 40 else "top"
    return f"{band}_{half}"


def make_synthetic_q2_parquets(processed_dir: Path) -> tuple[list[int], pd.DataFrame]:
    """Write synthetic Q2 labels + flat-event + spatial-event parquets.

    Signal: probability of label=1 grows with ``ball_x`` (closer to opp goal),
    so a logistic model should easily beat the ~10% base rate.
    """
    rng = np.random.default_rng(SYNTHETIC_Q2_RNG_SEED)
    rows = []
    for match_id in range(1, SYNTHETIC_Q2_N_MATCHES + 1):
        for evt in range(SYNTHETIC_Q2_EVENTS_PER_MATCH):
            ball_x = float(rng.uniform(0, 120))
            ball_y = float(rng.uniform(0, 80))
            dist_to_goal = float(np.hypot(120 - ball_x, 40 - ball_y))
            angle_to_goal = float(np.arctan2(40 - ball_y, 120 - ball_x))
            etype = SYNTHETIC_Q2_ACTION_TYPES[int(rng.integers(0, 4))]
            zone = _zone_for(ball_x, ball_y)
            spatial = {col: float(rng.normal(loc=2.0, scale=1.0)) for col in SPATIAL_COLUMNS}

            logit = -3.5 + 0.04 * ball_x + 0.5 * (etype == "Shot")
            prob = 1.0 / (1.0 + np.exp(-logit))
            label = int(rng.uniform() < prob)

            rows.append(
                {
                    "match_id": match_id,
                    "event_uuid": f"m{match_id}-e{evt}",
                    "type": etype,
                    "label": label,
                    "ball_x": ball_x,
                    "ball_y": ball_y,
                    "dist_to_goal": dist_to_goal,
                    "angle_to_goal": angle_to_goal,
                    "is_pass": int(etype == "Pass"),
                    "is_carry": int(etype == "Carry"),
                    "is_dribble": int(etype == "Dribble"),
                    "is_shot": int(etype == "Shot"),
                    "zone": zone,
                    **spatial,
                }
            )
    full = pd.DataFrame(rows)

    labels = full[["match_id", "event_uuid", "type", "label"]].copy()
    flat_cols = ["match_id", "event_uuid", "ball_x", "ball_y", "dist_to_goal",
                 "angle_to_goal", "is_pass", "is_carry", "is_dribble", "is_shot", "zone"]
    flat = full[flat_cols].copy()
    spatial_cols = ["match_id", "event_uuid"] + list(SPATIAL_COLUMNS)
    spatial = full[spatial_cols].copy()

    nan_idx = rng.choice(len(spatial), size=len(spatial) // 5, replace=False)
    spatial.loc[nan_idx, list(SPATIAL_COLUMNS)] = np.nan

    processed_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(processed_dir / "labels_q2_shot_in_n.parquet", index=False)
    flat.to_parquet(processed_dir / "features_flat_event.parquet", index=False)
    spatial.to_parquet(processed_dir / "features_spatial_event.parquet", index=False)

    match_ids = sorted(full["match_id"].unique().tolist())
    return match_ids, full


@pytest.fixture
def synthetic_q2(tmp_path: Path) -> tuple[Path, MatchSplits]:
    """Synthetic Q2 parquet dir + a deterministic match-level split."""
    processed = tmp_path / "processed"
    match_ids, _ = make_synthetic_q2_parquets(processed)
    splits = split_matches(match_ids, val_size=0.25, test_size=0.25, seed=2)
    return processed, splits


@pytest.fixture
def synthetic_q2_dirs(tmp_path: Path) -> tuple[Path, MatchSplits, Path, Path]:
    """Synthetic Q2 parquets + split + artifact dir + metrics CSV path."""
    processed = tmp_path / "processed"
    match_ids, _ = make_synthetic_q2_parquets(processed)
    splits = split_matches(match_ids, val_size=0.25, test_size=0.25, seed=2)
    artifacts = tmp_path / "artifacts"
    metrics_csv = tmp_path / "reports" / "metrics_log.csv"
    return processed, splits, artifacts, metrics_csv
