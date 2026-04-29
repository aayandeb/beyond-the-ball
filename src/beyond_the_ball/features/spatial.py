"""Spatial features derived from StatsBomb 360 freeze-frames.

For each event row with a ``freeze_frame`` (list of ``{teammate, actor, keeper,
location}`` dicts) we compute six features that describe the local pressure
and structure around the ball. Events whose freeze-frame has fewer than
``MIN_VISIBLE_PLAYERS`` entries are kept in the output but with NaN feature
values, so the per-event row count always matches the input.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from .flat import _coerce_event_uuid, _location_xy

# Goal posts and triangle constants (StatsBomb pitch 120x80).
GOAL_LEFT_POST: tuple[float, float] = (120.0, 36.0)
GOAL_RIGHT_POST: tuple[float, float] = (120.0, 44.0)

# Spec radii.
RADIUS_10M: float = 10.0
RADIUS_15M: float = 15.0

# Drop-policy threshold (PLAN §5 / AGENTS.md).
MIN_VISIBLE_PLAYERS: int = 18

# Q1 aggregate window: mean over the last N events of a possession.
Q1_WINDOW: int = 3

SPATIAL_COLUMNS: tuple[str, ...] = (
    "teammates_within_10m",
    "opponents_within_10m",
    "numerical_superiority_15m",
    "defensive_compactness",
    "opponents_between_ball_and_goal",
    "nearest_opponent_distance",
)


def point_in_triangle(
    p: tuple[float, float],
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
) -> bool:
    """Return True if point ``p`` lies inside triangle ``abc``.

    Uses the sign-of-cross-products method: a point is inside iff it is on the
    same side of each of the three edges. Boundary points count as inside.
    """
    def _sign(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

    d1 = _sign(p, a, b)
    d2 = _sign(p, b, c)
    d3 = _sign(p, c, a)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


def _player_locations(frame: Iterable[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return arrays (xy, is_teammate, is_actor) for one freeze-frame."""
    xs, ys, mates, actors = [], [], [], []
    for p in frame:
        loc = p.get("location")
        if loc is None:
            continue
        try:
            xs.append(float(loc[0]))
            ys.append(float(loc[1]))
        except (TypeError, IndexError, ValueError):
            continue
        mates.append(bool(p.get("teammate", False)))
        actors.append(bool(p.get("actor", False)))
    xy = np.column_stack([xs, ys]) if xs else np.zeros((0, 2))
    return xy, np.array(mates, dtype=bool), np.array(actors, dtype=bool)


def _is_valid_frame(frame: object) -> bool:
    if frame is None:
        return False
    try:
        return len(frame) > 0
    except TypeError:
        return False


def _compute_spatial_for_event(
    ball_xy: tuple[float, float],
    frame: Iterable[dict],
    n_visible: float,
) -> dict[str, float]:
    """Compute the 6 spatial features for one event. Returns NaNs if invalid."""
    nan_row = {col: np.nan for col in SPATIAL_COLUMNS}

    if not _is_valid_frame(frame):
        return nan_row
    if not (n_visible is not None and not np.isnan(n_visible) and n_visible >= MIN_VISIBLE_PLAYERS):
        return nan_row
    if np.isnan(ball_xy[0]) or np.isnan(ball_xy[1]):
        return nan_row

    xy, is_mate, is_actor = _player_locations(frame)
    if len(xy) == 0:
        return nan_row

    # Distances from each visible player to the ball.
    dx = xy[:, 0] - ball_xy[0]
    dy = xy[:, 1] - ball_xy[1]
    dist = np.sqrt(dx * dx + dy * dy)

    # Teammates exclude the ball carrier (actor); opponents are simply non-teammates.
    teammates_mask = is_mate & ~is_actor
    opponents_mask = ~is_mate

    teammates_10 = int(np.sum(teammates_mask & (dist <= RADIUS_10M)))
    opponents_10 = int(np.sum(opponents_mask & (dist <= RADIUS_10M)))
    teammates_15 = int(np.sum(teammates_mask & (dist <= RADIUS_15M)))
    opponents_15 = int(np.sum(opponents_mask & (dist <= RADIUS_15M)))

    if np.any(opponents_mask):
        opp_x = xy[opponents_mask, 0]
        compactness = float(np.std(opp_x))
        nearest_opp = float(np.min(dist[opponents_mask]))
    else:
        compactness = np.nan
        nearest_opp = np.nan

    # Count opponents inside triangle (ball, left post, right post).
    in_tri = 0
    if np.any(opponents_mask):
        for px, py in xy[opponents_mask]:
            if point_in_triangle((px, py), ball_xy, GOAL_LEFT_POST, GOAL_RIGHT_POST):
                in_tri += 1

    return {
        "teammates_within_10m": float(teammates_10),
        "opponents_within_10m": float(opponents_10),
        "numerical_superiority_15m": float(teammates_15 - opponents_15),
        "defensive_compactness": compactness,
        "opponents_between_ball_and_goal": float(in_tri),
        "nearest_opponent_distance": nearest_opp,
    }


def spatial_event_features(events_df: pd.DataFrame) -> pd.DataFrame:
    """Per-event spatial features keyed by ``(match_id, event_uuid)``.

    Output rows: one per input event (no row dropping). Events that fail the
    visibility / freeze-frame checks get NaN for all 6 spatial columns.
    """
    required = {"match_id", "location", "freeze_frame"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Events table missing required columns: {sorted(missing)}")

    has_n_visible = "n_visible_players" in events_df.columns

    records: list[dict[str, float]] = []
    for i in range(len(events_df)):
        row = events_df.iloc[i]
        ball_xy = _location_xy(row["location"])
        frame = row["freeze_frame"]
        n_vis = float(row["n_visible_players"]) if has_n_visible and pd.notna(row["n_visible_players"]) else (
            len(frame) if _is_valid_frame(frame) else 0.0
        )
        records.append(_compute_spatial_for_event(ball_xy, frame, n_vis))

    out = pd.DataFrame(records)
    out.insert(0, "match_id", events_df["match_id"].to_numpy())
    out.insert(1, "event_uuid", _coerce_event_uuid(events_df).to_numpy())
    return out.reset_index(drop=True)


def spatial_q1_possession_aggregate(
    per_event_features: pd.DataFrame,
    events_df: pd.DataFrame,
    *,
    window: int = Q1_WINDOW,
) -> pd.DataFrame:
    """Aggregate spatial features over the last ``window`` events of each possession.

    Returns a DataFrame keyed by ``(match_id, possession)`` with one column per
    spatial feature, suffixed ``_mean{window}`` (e.g. ``_mean3``). Possessions
    where every event in the window has NaN spatial features yield NaN for that
    column.
    """
    required = {"match_id", "possession"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Events table missing required columns: {sorted(missing)}")

    work = per_event_features.copy()
    work["possession"] = events_df["possession"].to_numpy()

    if "index" in events_df.columns:
        work["_order"] = events_df["index"].to_numpy()
    else:
        work["_order"] = np.arange(len(events_df))
    if "period" in events_df.columns:
        work["_period"] = events_df["period"].astype(float).to_numpy()
    else:
        work["_period"] = 1.0

    # Sort so that "last N events" is well-defined when a possession spans periods.
    work = work.sort_values(
        ["match_id", "possession", "_period", "_order"], kind="stable"
    )

    grouped = work.groupby(["match_id", "possession"], sort=False, dropna=False)
    last_n = grouped.tail(window)

    agg = last_n.groupby(["match_id", "possession"], sort=False, dropna=False)[
        list(SPATIAL_COLUMNS)
    ].mean().reset_index()

    rename = {col: f"{col}_mean{window}" for col in SPATIAL_COLUMNS}
    return agg.rename(columns=rename)


def build_spatial_feature_tables(
    canonical_path: str | Path = "data/interim/events_360_open_play.parquet",
    output_dir: str | Path = "data/processed",
) -> dict[str, pd.DataFrame]:
    """Compute event-level + Q1-possession-aggregated spatial features and persist as parquet."""
    events = pd.read_parquet(Path(canonical_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    per_event = spatial_event_features(events)
    per_event_path = out_dir / "features_spatial_event.parquet"
    per_event.to_parquet(per_event_path, index=False)

    per_possession = spatial_q1_possession_aggregate(per_event, events)
    per_possession_path = out_dir / "features_spatial_possession.parquet"
    per_possession.to_parquet(per_possession_path, index=False)

    return {"event": per_event, "possession": per_possession}
