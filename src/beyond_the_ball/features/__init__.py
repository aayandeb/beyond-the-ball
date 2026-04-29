"""Feature engineering and label utilities."""

from .flat import (
    ACTION_TYPES,
    GOAL_X,
    GOAL_Y,
    build_flat_feature_tables,
    flat_event_features,
    pitch_zone,
    possession_aggregates,
)
from .labels import (
    ON_BALL_EVENT_TYPES,
    build_label_tables,
    label_q1_possession_terminal,
    label_q2_shot_in_n,
)
from .spatial import (
    MIN_VISIBLE_PLAYERS,
    SPATIAL_COLUMNS,
    build_spatial_feature_tables,
    point_in_triangle,
    spatial_event_features,
    spatial_q1_possession_aggregate,
)

__all__ = [
    "ACTION_TYPES",
    "GOAL_X",
    "GOAL_Y",
    "MIN_VISIBLE_PLAYERS",
    "ON_BALL_EVENT_TYPES",
    "SPATIAL_COLUMNS",
    "build_flat_feature_tables",
    "build_label_tables",
    "build_spatial_feature_tables",
    "flat_event_features",
    "label_q1_possession_terminal",
    "label_q2_shot_in_n",
    "pitch_zone",
    "point_in_triangle",
    "possession_aggregates",
    "spatial_event_features",
    "spatial_q1_possession_aggregate",
]
