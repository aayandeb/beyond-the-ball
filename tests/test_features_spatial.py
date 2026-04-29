"""Tests for M5 spatial features (uses 360 freeze-frame data)."""

import math

import numpy as np
import pandas as pd

from beyond_the_ball.features.spatial import (
    MIN_VISIBLE_PLAYERS,
    point_in_triangle,
    spatial_event_features,
    spatial_q1_possession_aggregate,
)


def _player(loc, teammate, actor=False, keeper=False):
    # Helper to build one freeze-frame entry.
    return {
        "teammate": teammate,
        "actor": actor,
        "keeper": keeper,
        "location": np.array(loc, dtype=float),
    }


def _event_row(match_id, possession, ball_xy, freeze_frame, event_id, index=0):
    n_visible = len(freeze_frame)
    return {
        "match_id": match_id,
        "possession": possession,
        "index": index,
        "id": event_id,
        "location": [float(ball_xy[0]), float(ball_xy[1])],
        "freeze_frame": freeze_frame,
        "n_visible_players": float(n_visible),
    }


def _pad_to_min_visible(frame):
    # Pad with far-away teammates so the event clears the MIN_VISIBLE_PLAYERS gate
    # without affecting any of the within-N-meters counts (placed at far corner).
    pad_count = max(0, MIN_VISIBLE_PLAYERS - len(frame))
    return frame + [_player([1.0, 1.0], teammate=True) for _ in range(pad_count)]


def test_point_in_triangle_basic():
    # Triangle ball=(60,40), posts (120,36) and (120,44).
    a, b, c = (60.0, 40.0), (120.0, 36.0), (120.0, 44.0)
    # On the x=100 vertical line between y=38 and y=42 -> inside.
    assert point_in_triangle((100.0, 40.0), a, b, c)
    # Far above the triangle -> outside.
    assert not point_in_triangle((100.0, 50.0), a, b, c)
    # Behind the ball (smaller x) -> outside.
    assert not point_in_triangle((50.0, 40.0), a, b, c)


def test_spatial_features_counts_within_radius():
    ball = (60.0, 40.0)
    # 2 teammates within 10m (one is the actor and must be excluded), 1 opponent within 10m.
    frame = [
        _player(ball, teammate=True, actor=True),       # actor — excluded from teammate counts
        _player([65.0, 40.0], teammate=True),           # 5m away teammate
        _player([55.0, 40.0], teammate=True),           # 5m away teammate
        _player([62.0, 41.0], teammate=False),          # ~2.2m opponent
        _player([100.0, 40.0], teammate=False),         # far opponent (40m)
    ]
    frame = _pad_to_min_visible(frame)

    events = pd.DataFrame([_event_row(1, 1, ball, frame, "e1")])
    out = spatial_event_features(events)
    row = out.iloc[0]

    assert row["teammates_within_10m"] == 2
    assert row["opponents_within_10m"] == 1
    # Within 15m: 2 teammates, 1 opponent -> superiority = 1.
    assert row["numerical_superiority_15m"] == 1


def test_spatial_features_nearest_opponent_and_compactness():
    ball = (60.0, 40.0)
    frame = [
        _player(ball, teammate=True, actor=True),
        _player([62.0, 40.0], teammate=False),    # nearest opp at 2m
        _player([100.0, 40.0], teammate=False),
        _player([90.0, 60.0], teammate=False),
    ]
    frame = _pad_to_min_visible(frame)
    events = pd.DataFrame([_event_row(1, 1, ball, frame, "e1")])
    out = spatial_event_features(events)
    row = out.iloc[0]

    assert math.isclose(row["nearest_opponent_distance"], 2.0, abs_tol=1e-6)
    expected_std = float(np.std([62.0, 100.0, 90.0]))
    assert math.isclose(row["defensive_compactness"], expected_std, abs_tol=1e-6)


def test_spatial_features_opponents_between_ball_and_goal():
    ball = (60.0, 40.0)
    frame = [
        _player(ball, teammate=True, actor=True),
        _player([90.0, 40.0], teammate=False),   # inside the ball->goal triangle
        _player([100.0, 39.0], teammate=False),  # inside (close to center)
        _player([100.0, 50.0], teammate=False),  # outside (above triangle)
        _player([50.0, 40.0], teammate=False),   # outside (behind ball)
    ]
    frame = _pad_to_min_visible(frame)
    events = pd.DataFrame([_event_row(1, 1, ball, frame, "e1")])
    out = spatial_event_features(events)
    row = out.iloc[0]

    assert row["opponents_between_ball_and_goal"] == 2


def test_spatial_features_drops_low_visibility():
    ball = (60.0, 40.0)
    # Only 5 visible -> below MIN_VISIBLE_PLAYERS -> NaN row.
    frame = [
        _player(ball, teammate=True, actor=True),
        _player([62.0, 40.0], teammate=False),
        _player([65.0, 40.0], teammate=True),
        _player([90.0, 40.0], teammate=False),
        _player([100.0, 50.0], teammate=False),
    ]
    events = pd.DataFrame([_event_row(1, 1, ball, frame, "e1")])
    out = spatial_event_features(events)
    row = out.iloc[0]

    # All 6 spatial feature columns should be NaN.
    for col in (
        "teammates_within_10m",
        "opponents_within_10m",
        "numerical_superiority_15m",
        "defensive_compactness",
        "opponents_between_ball_and_goal",
        "nearest_opponent_distance",
    ):
        assert pd.isna(row[col]), f"{col} should be NaN when visibility is low"


def test_spatial_q1_aggregate_uses_last_three():
    ball = (60.0, 40.0)
    # Build 4 events in the same possession, each with one opponent at varying distances.
    rows = []
    distances = [50.0, 5.0, 6.0, 7.0]  # last three: 5,6,7 -> mean nearest_opponent_distance = 6.0
    for i, d in enumerate(distances):
        frame = _pad_to_min_visible(
            [
                _player(ball, teammate=True, actor=True),
                _player([60.0 + d, 40.0], teammate=False),
            ]
        )
        rows.append(_event_row(1, 7, ball, frame, f"e{i}", index=i))

    events = pd.DataFrame(rows)
    per_event = spatial_event_features(events)
    agg = spatial_q1_possession_aggregate(per_event, events)
    row = agg.iloc[0]

    assert row["match_id"] == 1
    assert row["possession"] == 7
    assert math.isclose(row["nearest_opponent_distance_mean3"], 6.0, abs_tol=1e-6)
