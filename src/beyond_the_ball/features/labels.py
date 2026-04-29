"""Label functions for Q1 (possession terminal) and Q2 (shot-in-N)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
import pandas as pd

ON_BALL_EVENT_TYPES: tuple[str, ...] = ("Pass", "Carry", "Dribble", "Shot")
FINAL_THIRD_X: float = 80.0
SHOT_TYPE: str = "Shot"


def _extract_name(value: object) -> object:
    if isinstance(value, dict):
        return value.get("name")
    return value


def _location_x(value: object) -> float:
    if value is None:
        return np.nan
    try:
        return float(value[0])
    except (TypeError, IndexError, ValueError):
        return np.nan


def _coerce_event_uuid(events_df: pd.DataFrame) -> pd.Series:
    if "event_uuid" in events_df.columns:
        return events_df["event_uuid"].astype(str)
    if "id" in events_df.columns:
        return events_df["id"].astype(str)
    raise ValueError("Events table must contain 'event_uuid' or 'id'.")


def _resolve_team(events_df: pd.DataFrame) -> pd.Series:
    if "team" in events_df.columns:
        team = events_df["team"]
    elif "possession_team" in events_df.columns:
        team = events_df["possession_team"]
    else:
        raise ValueError("Events table must contain 'team' or 'possession_team'.")
    return team.map(_extract_name).astype("object")


def label_q1_possession_terminal(
    events_df: pd.DataFrame,
    *,
    min_events: int = 3,
    final_third_x: float = FINAL_THIRD_X,
) -> pd.DataFrame:
    """Classify each possession as ``shot`` / ``final_third`` / ``turnover``.

    Possessions with fewer than ``min_events`` events are excluded.
    """
    required = {"match_id", "possession", "type", "location"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Events table missing required columns: {sorted(missing)}")

    work = pd.DataFrame(
        {
            "match_id": events_df["match_id"].to_numpy(),
            "possession": events_df["possession"].to_numpy(),
            "_type": events_df["type"].map(_extract_name).to_numpy(),
            "_x": [_location_x(v) for v in events_df["location"].to_numpy()],
        }
    )

    grouped = work.groupby(["match_id", "possession"], sort=False, dropna=False)
    summary = grouped.agg(
        n_events=("_type", "size"),
        has_shot=("_type", lambda s: bool((s == SHOT_TYPE).any())),
        max_x=("_x", "max"),
    ).reset_index()

    summary = summary[summary["n_events"] >= min_events].copy()

    label = np.where(
        summary["has_shot"].to_numpy(),
        "shot",
        np.where(summary["max_x"].fillna(-np.inf).to_numpy() >= final_third_x, "final_third", "turnover"),
    )
    summary["label"] = label

    return summary[["match_id", "possession", "n_events", "label"]].reset_index(drop=True)


def _shot_in_next_n_within_group(
    types: np.ndarray,
    teams: np.ndarray,
    n: int,
) -> np.ndarray:
    is_shot = types == SHOT_TYPE
    labels = np.zeros(len(types), dtype=np.int8)
    size = len(types)
    for i in range(size):
        team_i = teams[i]
        end = min(i + 1 + n, size)
        for j in range(i + 1, end):
            if is_shot[j] and teams[j] == team_i:
                labels[i] = 1
                break
    return labels


def label_q2_shot_in_n(
    events_df: pd.DataFrame,
    *,
    n: int = 10,
    restrict_to_possession: bool = True,
    on_ball_types: Sequence[str] | Iterable[str] = ON_BALL_EVENT_TYPES,
) -> pd.DataFrame:
    """Label each on-ball event with whether the same team shoots in the next ``n`` events.

    The look-ahead is restricted to the same possession by default (recommended
    in the project plan); set ``restrict_to_possession=False`` to allow it to
    span possessions within the same match.
    """
    required = {"match_id", "type"}
    missing = required - set(events_df.columns)
    if missing:
        raise ValueError(f"Events table missing required columns: {sorted(missing)}")
    if restrict_to_possession and "possession" not in events_df.columns:
        raise ValueError("'possession' column required when restrict_to_possession=True")

    on_ball_set = set(on_ball_types)

    work = pd.DataFrame(
        {
            "match_id": events_df["match_id"].to_numpy(),
            "possession": events_df["possession"].to_numpy()
            if "possession" in events_df.columns
            else np.zeros(len(events_df), dtype=np.int64),
            "_type": events_df["type"].map(_extract_name).to_numpy(),
            "_team": _resolve_team(events_df).to_numpy(),
            "event_uuid": _coerce_event_uuid(events_df).to_numpy(),
        }
    )

    if "index" in events_df.columns:
        work["_order"] = events_df["index"].to_numpy()
    else:
        work["_order"] = np.arange(len(events_df))

    work = work.sort_values(["match_id", "_order"], kind="stable").reset_index(drop=True)

    group_cols = ["match_id", "possession"] if restrict_to_possession else ["match_id"]

    labels = np.zeros(len(work), dtype=np.int8)
    for _, group in work.groupby(group_cols, sort=False, dropna=False):
        idx = group.index.to_numpy()
        labels[idx] = _shot_in_next_n_within_group(
            types=group["_type"].to_numpy(),
            teams=group["_team"].to_numpy(),
            n=n,
        )

    work["label"] = labels
    on_ball_mask = work["_type"].isin(on_ball_set)
    out = work.loc[on_ball_mask, ["match_id", "event_uuid", "_type", "label"]].rename(
        columns={"_type": "type"}
    )
    return out.reset_index(drop=True)


def build_label_tables(
    canonical_path: str | Path = "data/interim/events_360_open_play.parquet",
    output_dir: str | Path = "data/processed",
    *,
    min_events: int = 3,
    n: int = 10,
    restrict_to_possession: bool = True,
) -> dict[str, pd.DataFrame]:
    """Compute Q1 and Q2 labels from the canonical table and persist as parquet."""
    events = pd.read_parquet(Path(canonical_path))
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    q1 = label_q1_possession_terminal(events, min_events=min_events)
    q1_path = out_dir / "labels_q1_possession.parquet"
    q1.to_parquet(q1_path, index=False)

    q2 = label_q2_shot_in_n(
        events,
        n=n,
        restrict_to_possession=restrict_to_possession,
    )
    q2_path = out_dir / "labels_q2_shot_in_n.parquet"
    q2.to_parquet(q2_path, index=False)

    return {"q1": q1, "q2": q2}
