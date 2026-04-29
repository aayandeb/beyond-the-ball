"""Flat (non-spatial) features: per-event geometry + per-possession aggregates."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

# Pitch + goal constants. StatsBomb pitch is 120 x 80; opponent goal is at x=120, y in [36, 44].
PITCH_LENGTH: float = 120.0
PITCH_WIDTH: float = 80.0
GOAL_X: float = 120.0
GOAL_Y: float = 40.0

# Action types we one-hot encode. Anything else -> all zeros.
ACTION_TYPES: tuple[str, ...] = ("Pass", "Carry", "Dribble", "Shot")
PASS_TYPE: str = "Pass"

# 6 zones: 3 x-bands (defensive/middle/attacking) x 2 y-halves (bottom/top).
# x splits at 40 and 80; y split at 40.
_DEF_THIRD_X: float = 40.0
_ATT_THIRD_X: float = 80.0
_HALF_Y: float = 40.0


def _extract_name(value: object) -> object:
    # StatsBomb often nests {"id": ..., "name": ...} dicts. Pull the name out.
    if isinstance(value, dict):
        return value.get("name")
    return value


def _location_xy(value: object) -> tuple[float, float]:
    # Locations are [x, y] lists. Return NaNs if missing/malformed.
    if value is None:
        return (np.nan, np.nan)
    try:
        return (float(value[0]), float(value[1]))
    except (TypeError, IndexError, ValueError):
        return (np.nan, np.nan)


def _coerce_event_uuid(events_df: pd.DataFrame) -> pd.Series:
    if "event_uuid" in events_df.columns:
        return events_df["event_uuid"].astype(str)
    if "id" in events_df.columns:
        return events_df["id"].astype(str)
    raise ValueError("Events table must contain 'event_uuid' or 'id'.")


def pitch_zone(x: float, y: float) -> str:
    """Return one of 6 zone labels based on (x, y) on a 120x80 pitch."""
    if np.isnan(x) or np.isnan(y):
        return "unknown"
    band = "def" if x < _DEF_THIRD_X else ("mid" if x < _ATT_THIRD_X else "att")
    half = "bottom" if y < _HALF_Y else "top"
    return f"{band}_{half}"


def flat_event_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Per-event flat features keyed by ``(match_id, event_uuid)``.

    Columns: ball_x, ball_y, dist_to_goal, angle_to_goal, is_<action>, zone.
    NaN policy: numeric features get NaN if location is missing; ``zone`` becomes
    ``"unknown"``. Action one-hots are always 0/1 (never NaN).
    """
    required = {"match_id", "type", "location"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Events table missing required columns: {sorted(missing)}")

    n = len(events_df)
    locs = [_location_xy(v) for v in events_df["location"].to_numpy()]
    xs = np.array([p[0] for p in locs], dtype=float)
    ys = np.array([p[1] for p in locs], dtype=float)

    # Geometry: distance and angle from ball position to goal center.
    dx = GOAL_X - xs
    dy = GOAL_Y - ys
    dist = np.sqrt(dx * dx + dy * dy)
    angle = np.arctan2(dy, dx)

    types = pd.Series(events_df["type"]).map(_extract_name).to_numpy()

    out = pd.DataFrame(
        {
            "match_id": events_df["match_id"].to_numpy(),
            "event_uuid": _coerce_event_uuid(events_df).to_numpy(),
            "ball_x": xs,
            "ball_y": ys,
            "dist_to_goal": dist,
            "angle_to_goal": angle,
        }
    )

    # One-hot for the recognized action types only. Unknown -> all zeros.
    for action in ACTION_TYPES:
        out[f"is_{action.lower()}"] = (types == action).astype(np.int8)

    out["zone"] = [pitch_zone(x, y) for x, y in zip(xs, ys)]

    assert len(out) == n  # sanity: shape preserved.
    return out.reset_index(drop=True)


def possession_aggregates(
    events_df: pd.DataFrame,
    *,
    on_ball_types: Sequence[str] | Iterable[str] = ACTION_TYPES,
) -> pd.DataFrame:
    """Per-possession aggregates keyed by ``(match_id, possession)``.

    Columns: n_events, duration_s, x_progression, n_passes, mean_x, max_x.
    """
    required = {"match_id", "possession", "type", "location", "minute", "second"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Events table missing required columns: {sorted(missing)}")

    locs = [_location_xy(v) for v in events_df["location"].to_numpy()]
    xs = np.array([p[0] for p in locs], dtype=float)

    # Absolute event time in seconds. StatsBomb restarts ``minute`` each period
    # (period 1 starts at 0, period 2 at 45, etc.) and possessions occasionally
    # span the half-time boundary, so we offset by period to keep ``t`` monotonic.
    minutes = events_df["minute"].astype(float).to_numpy()
    seconds = events_df["second"].astype(float).to_numpy()
    if "period" in events_df.columns:
        periods = events_df["period"].astype(float).to_numpy()
    else:
        periods = np.ones(len(events_df), dtype=float)
    _PERIOD_OFFSET_S = 3600.0  # 60 minutes — comfortably larger than any half + stoppage.
    t = periods * _PERIOD_OFFSET_S + minutes * 60.0 + seconds

    types = pd.Series(events_df["type"]).map(_extract_name).to_numpy()

    work = pd.DataFrame(
        {
            "match_id": events_df["match_id"].to_numpy(),
            "possession": events_df["possession"].to_numpy(),
            "_x": xs,
            "_t": t,
            "_type": types,
            "_is_pass": (types == PASS_TYPE).astype(np.int64),
        }
    )

    # Order within possession by (period, index) so half-time spanning possessions stay correct.
    if "index" in events_df.columns:
        work["_order"] = events_df["index"].to_numpy()
    else:
        work["_order"] = np.arange(len(events_df))
    work["_period"] = periods
    work = work.sort_values(
        ["match_id", "possession", "_period", "_order"], kind="stable"
    )

    grouped = work.groupby(["match_id", "possession"], sort=False, dropna=False)

    summary = grouped.agg(
        n_events=("_type", "size"),
        n_passes=("_is_pass", "sum"),
        mean_x=("_x", "mean"),
        max_x=("_x", "max"),
        first_x=("_x", "first"),
        last_x=("_x", "last"),
        first_t=("_t", "first"),
        last_t=("_t", "last"),
    ).reset_index()

    summary["duration_s"] = summary["last_t"] - summary["first_t"]
    summary["x_progression"] = summary["last_x"] - summary["first_x"]

    cols = [
        "match_id",
        "possession",
        "n_events",
        "duration_s",
        "x_progression",
        "n_passes",
        "mean_x",
        "max_x",
    ]
    return summary[cols].reset_index(drop=True)


def build_flat_feature_tables(
    canonical_path: str | Path = "data/interim/events_360_open_play.parquet",
    output_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    """Compute event-level + possession-level flat features and persist as parquet."""
    events = pd.read_parquet(Path(canonical_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_event = flat_event_features(events)
    per_event_path = out_dir / "features_flat_event.parquet"
    per_event.to_parquet(per_event_path, index=False)

    per_possession = possession_aggregates(events)
    per_possession_path = out_dir / "features_flat_possession.parquet"
    per_possession.to_parquet(per_possession_path, index=False)

    return {"event": per_event, "possession": per_possession}
