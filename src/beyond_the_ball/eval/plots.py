"""Plot helpers for visual sanity checks."""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from mplsoccer import Pitch


def plot_freeze_frame_event(
    canonical_df: pd.DataFrame,
    event_uuid: str,
    *,
    title: str | None = None,
):
    """Plot one event with its freeze-frame players on a StatsBomb pitch."""
    row = canonical_df.loc[canonical_df["event_uuid"] == event_uuid]
    if row.empty:
        raise ValueError(f"Event UUID not found: {event_uuid}")

    event = row.iloc[0]
    ball_loc = event.get("location")
    freeze_frame = event.get("freeze_frame")
    try:
        ball_loc = list(ball_loc)
    except TypeError as exc:
        raise ValueError("Event row does not contain a valid 'location'.") from exc
    if len(ball_loc) != 2:
        raise ValueError("Event row 'location' must have two coordinates.")
    try:
        freeze_frame = list(freeze_frame)
    except TypeError as exc:
        raise ValueError("Event row does not contain a valid 'freeze_frame'.") from exc

    pitch = Pitch(pitch_type="statsbomb", line_zorder=2)
    fig, ax = pitch.draw(figsize=(10, 6))

    attackers_x, attackers_y = [], []
    defenders_x, defenders_y = [], []
    for player in freeze_frame:
        loc = player.get("location")
        try:
            loc = list(loc)
        except TypeError:
            continue
        if len(loc) != 2:
            continue
        if player.get("teammate"):
            attackers_x.append(loc[0])
            attackers_y.append(loc[1])
        else:
            defenders_x.append(loc[0])
            defenders_y.append(loc[1])

    if attackers_x:
        pitch.scatter(attackers_x, attackers_y, c="#2ca02c", s=70, ax=ax, label="Teammates")
    if defenders_x:
        pitch.scatter(defenders_x, defenders_y, c="#d62728", s=70, ax=ax, label="Opponents")

    pitch.scatter([ball_loc[0]], [ball_loc[1]], c="#1f77b4", s=110, ax=ax, label="Ball event")
    pitch.lines(ball_loc[0], ball_loc[1], 120, 40, lw=2, color="#1f77b4", ax=ax)

    ax.legend(loc="upper left")
    ax.set_title(title or f"Freeze-frame sanity check: {event_uuid}")
    return fig, ax

