"""Tests for M4 flat features."""

import math

import pandas as pd

from beyond_the_ball.features.flat import (
    ACTION_TYPES,
    flat_event_features,
    pitch_zone,
    possession_aggregates,
)


def _event(match_id, index, possession, team, etype, x, y, minute=0, second=0, event_id=None):
    return {
        "match_id": match_id,
        "index": index,
        "possession": possession,
        "team": team,
        "type": etype,
        "location": [float(x), float(y)],
        "minute": minute,
        "second": second,
        "id": event_id or f"m{match_id}-i{index}",
    }


def test_pitch_zone_six_zones():
    # 3 x-bands (def/mid/att) x 2 y-halves (bottom/top), pitch 120x80, halves split at y=40.
    assert pitch_zone(10, 10) == "def_bottom"
    assert pitch_zone(10, 70) == "def_top"
    assert pitch_zone(50, 20) == "mid_bottom"
    assert pitch_zone(50, 60) == "mid_top"
    assert pitch_zone(100, 10) == "att_bottom"
    assert pitch_zone(100, 70) == "att_top"


def test_flat_event_features_distance_and_angle_to_goal():
    events = pd.DataFrame(
        [
            _event(1, 1, 1, "A", "Pass", 60, 40, event_id="e1"),
            _event(1, 2, 1, "A", "Shot", 114, 40, event_id="e2"),
        ]
    )
    out = flat_event_features(events)
    by_id = {row["event_uuid"]: row for _, row in out.iterrows()}

    assert math.isclose(by_id["e1"]["dist_to_goal"], 60.0, abs_tol=1e-6)
    assert math.isclose(by_id["e1"]["angle_to_goal"], 0.0, abs_tol=1e-6)
    assert math.isclose(by_id["e2"]["dist_to_goal"], 6.0, abs_tol=1e-6)


def test_flat_event_features_action_one_hot_and_zone():
    events = pd.DataFrame(
        [
            _event(1, 1, 1, "A", "Pass", 60, 40, event_id="e1"),
            _event(1, 2, 1, "A", "Carry", 70, 30, event_id="e2"),
            _event(1, 3, 1, "A", "Miscontrol", 55, 50, event_id="e3"),
        ]
    )
    out = flat_event_features(events)
    by_id = {row["event_uuid"]: row for _, row in out.iterrows()}

    for action in ACTION_TYPES:
        assert f"is_{action.lower()}" in out.columns

    assert by_id["e1"]["is_pass"] == 1
    assert by_id["e1"]["is_carry"] == 0
    assert by_id["e2"]["is_carry"] == 1
    # Unrecognized action -> all zeros, no NaN.
    assert by_id["e3"]["is_pass"] == 0
    assert by_id["e3"]["is_carry"] == 0

    assert by_id["e1"]["zone"] == "mid_top"
    assert by_id["e2"]["zone"] == "mid_bottom"
    assert by_id["e3"]["zone"] == "mid_top"


def test_flat_event_features_keys_and_no_nans():
    events = pd.DataFrame(
        [
            _event(1, 1, 1, "A", "Pass", 60, 40, event_id="e1"),
            _event(2, 1, 5, "B", "Carry", 70, 40, event_id="e2"),
        ]
    )
    out = flat_event_features(events)
    assert list(out.columns)[:2] == ["match_id", "event_uuid"]
    assert not out.drop(columns=["zone"]).isna().any().any()


def test_possession_aggregates_basic():
    events = pd.DataFrame(
        [
            _event(1, 1, 1, "A", "Pass", 30, 40, minute=0, second=0),
            _event(1, 2, 1, "A", "Pass", 50, 40, minute=0, second=5),
            _event(1, 3, 1, "A", "Carry", 70, 40, minute=0, second=10),
            _event(1, 4, 1, "A", "Shot", 110, 40, minute=0, second=12),
        ]
    )
    out = possession_aggregates(events)
    row = out.iloc[0]

    assert row["match_id"] == 1
    assert row["possession"] == 1
    assert row["n_events"] == 4
    assert math.isclose(row["duration_s"], 12.0, abs_tol=1e-6)
    assert math.isclose(row["x_progression"], 80.0, abs_tol=1e-6)
    assert row["n_passes"] == 2
    assert math.isclose(row["mean_x"], 65.0, abs_tol=1e-6)
    assert math.isclose(row["max_x"], 110.0, abs_tol=1e-6)


def test_possession_aggregates_no_nans_multi_match():
    events = pd.DataFrame(
        [
            _event(1, 1, 1, "A", "Pass", 30, 40),
            _event(1, 2, 1, "A", "Pass", 40, 40),
            _event(2, 1, 5, "B", "Pass", 60, 40),
            _event(2, 2, 5, "B", "Carry", 70, 40),
        ]
    )
    out = possession_aggregates(events)
    assert len(out) == 2
    assert not out.isna().any().any()
