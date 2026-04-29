"""Utilities for building a canonical events + 360 table."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

SET_PIECE_PLAY_PATTERNS = {
    "From Corner",
    "From Free Kick",
    "From Goal Kick",
    "From Kick Off",
    "From Throw In",
    "From Keeper",
}

SET_PIECE_EVENT_TYPES = {
    "Corner",
    "Free Kick",
    "Throw-in",
    "Kick Off",
    "Goal Kick",
    "Penalty",
}


def _extract_name(value: object) -> object:
    """Return a readable name from StatsBomb nested values."""
    if isinstance(value, dict):
        return value.get("name")
    return value


def _event_uuid_column(events_df: pd.DataFrame) -> pd.Series:
    """Extract the event UUID from the events table."""
    if "id" in events_df.columns:
        return events_df["id"].astype(str)
    if "event_uuid" in events_df.columns:
        return events_df["event_uuid"].astype(str)
    raise ValueError("Events table must contain either 'id' or 'event_uuid'.")


def join_events_with_360(events_df: pd.DataFrame, frames_df: pd.DataFrame) -> pd.DataFrame:
    """Join one match's events to 360 freeze-frame rows."""
    events = events_df.copy()
    frames = frames_df.copy()

    if "event_uuid" not in frames.columns:
        raise ValueError("Frames table must contain 'event_uuid'.")

    events["event_uuid"] = _event_uuid_column(events)
    frames["event_uuid"] = frames["event_uuid"].astype(str)

    if "n_visible_players" not in frames.columns and "freeze_frame" in frames.columns:
        frames["n_visible_players"] = frames["freeze_frame"].apply(
            lambda row: len(row) if isinstance(row, list) else 0
        )

    canonical = events.merge(
        frames,
        how="left",
        on="event_uuid",
        suffixes=("", "_360"),
        validate="one_to_one",
    )
    return canonical


def filter_open_play(canonical_df: pd.DataFrame) -> pd.DataFrame:
    """Drop set-piece sequences based on play pattern + event type."""
    df = canonical_df.copy()

    if "play_pattern" in df.columns:
        play_pattern = df["play_pattern"].map(_extract_name)
    elif "play_pattern_name" in df.columns:
        play_pattern = df["play_pattern_name"].map(_extract_name)
    else:
        play_pattern = pd.Series([None] * len(df), index=df.index)

    if "type" in df.columns:
        event_type = df["type"].map(_extract_name)
    elif "type_name" in df.columns:
        event_type = df["type_name"].map(_extract_name)
    else:
        event_type = pd.Series([None] * len(df), index=df.index)

    is_regular_play = play_pattern.eq("Regular Play")
    is_not_set_piece_type = ~event_type.isin(SET_PIECE_EVENT_TYPES)

    # If play pattern is missing, keep the row unless event type is explicit set piece.
    keep_mask = (is_regular_play | play_pattern.isna()) & is_not_set_piece_type
    keep_mask &= ~play_pattern.isin(SET_PIECE_PLAY_PATTERNS)

    return df.loc[keep_mask].reset_index(drop=True)


def _iter_match_ids(events_dir: Path) -> Iterable[int]:
    for path in sorted(events_dir.glob("*.parquet")):
        try:
            yield int(path.stem)
        except ValueError:
            continue


def build_canonical_table(
    raw_dir: str | Path = "data/raw",
    output_path: str | Path = "data/interim/events_360_open_play.parquet",
) -> pd.DataFrame:
    """Build and persist the canonical open-play events + 360 parquet table."""
    raw_path = Path(raw_dir)
    events_dir = raw_path / "events"
    frames_dir = raw_path / "frames"

    if not events_dir.exists():
        raise FileNotFoundError(f"Missing events directory: {events_dir}")
    if not frames_dir.exists():
        raise FileNotFoundError(f"Missing frames directory: {frames_dir}")

    per_match_tables: list[pd.DataFrame] = []
    for match_id in _iter_match_ids(events_dir):
        events_path = events_dir / f"{match_id}.parquet"
        frames_path = frames_dir / f"{match_id}.parquet"
        if not frames_path.exists():
            continue

        events_df = pd.read_parquet(events_path)
        frames_df = pd.read_parquet(frames_path)

        canonical = join_events_with_360(events_df=events_df, frames_df=frames_df)
        canonical["match_id"] = canonical.get("match_id", match_id)
        canonical = filter_open_play(canonical)
        per_match_tables.append(canonical)

    if not per_match_tables:
        raise RuntimeError("No canonical tables were produced from raw data.")

    final_df = pd.concat(per_match_tables, ignore_index=True)
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_parquet(target_path, index=False)
    return final_df


def load_canonical_table(
    path: str | Path = "data/interim/events_360_open_play.parquet",
) -> pd.DataFrame:
    """Load the canonical M2 parquet table."""
    return pd.read_parquet(Path(path))


