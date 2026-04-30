"""Dataset assembly for Q1 (per-possession, multiclass) and Q2 (per-event, binary).

Both tasks join feature parquets onto labels, apply a match-level split, and
return numeric ``(X, y)`` arrays plus the fitted preprocessor. Imputation
(median) and scaling (StandardScaler) are fit on **train only** — never on
val or test.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ..data.splits import MatchSplits
from ..features.flat import build_flat_feature_tables
from ..features.spatial import SPATIAL_COLUMNS, build_spatial_feature_tables

DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_CANONICAL_PATH = Path("data/interim/events_360_open_play.parquet")

Q1_FLAT_FEATURES: tuple[str, ...] = (
    "n_events",
    "duration_s",
    "x_progression",
    "n_passes",
    "mean_x",
    "max_x",
)
Q1_SPATIAL_FEATURES: tuple[str, ...] = tuple(f"{c}_mean3" for c in SPATIAL_COLUMNS)

# Q2 per-event features.
Q2_NUMERIC_FLAT_FEATURES: tuple[str, ...] = (
    "ball_x",
    "ball_y",
    "dist_to_goal",
    "angle_to_goal",
    "is_pass",
    "is_carry",
    "is_dribble",
    "is_shot",
)
Q2_ZONE_VALUES: tuple[str, ...] = (
    "def_bottom",
    "def_top",
    "mid_bottom",
    "mid_top",
    "att_bottom",
    "att_top",
    "unknown",
)
Q2_ZONE_FEATURES: tuple[str, ...] = tuple(f"zone_{z}" for z in Q2_ZONE_VALUES)
Q2_FLAT_FEATURES: tuple[str, ...] = Q2_NUMERIC_FLAT_FEATURES + Q2_ZONE_FEATURES
Q2_SPATIAL_FEATURES: tuple[str, ...] = tuple(SPATIAL_COLUMNS)


@dataclass
class Q1SplitData:
    """Per-split feature matrix and label vector for Q1."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    classes: tuple[str, ...]
    preprocessor: Pipeline
    keys_train: pd.DataFrame  # match_id, possession
    keys_val: pd.DataFrame
    keys_test: pd.DataFrame


def ensure_feature_tables(
    canonical_path: str | Path = DEFAULT_CANONICAL_PATH,
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
) -> None:
    """Materialize flat + spatial feature parquets if any are missing."""
    processed = Path(processed_dir)
    canonical = Path(canonical_path)

    flat_event = processed / "features_flat_event.parquet"
    flat_poss = processed / "features_flat_possession.parquet"
    spat_event = processed / "features_spatial_event.parquet"
    spat_poss = processed / "features_spatial_possession.parquet"

    if not (flat_event.exists() and flat_poss.exists()):
        build_flat_feature_tables(canonical, processed)
    if not (spat_event.exists() and spat_poss.exists()):
        build_spatial_feature_tables(canonical, processed)


def load_q1_table(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    *,
    include_spatial: bool = True,
) -> pd.DataFrame:
    """Return the joined Q1 table: labels + flat (+ spatial) per possession."""
    processed = Path(processed_dir)
    labels = pd.read_parquet(processed / "labels_q1_possession.parquet")
    flat = pd.read_parquet(processed / "features_flat_possession.parquet")

    df = labels.merge(
        flat,
        on=["match_id", "possession"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_flat"),
    )
    if "n_events_flat" in df.columns:
        df = df.drop(columns=["n_events_flat"])  # both tables expose n_events.

    if include_spatial:
        spatial = pd.read_parquet(processed / "features_spatial_possession.parquet")
        df = df.merge(
            spatial,
            on=["match_id", "possession"],
            how="left",
            validate="one_to_one",
        )
    return df


def _build_preprocessor(*, scale: bool) -> Pipeline:
    steps: list[tuple[str, object]] = [
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps)


def prepare_q1_split(
    df: pd.DataFrame,
    splits: MatchSplits,
    *,
    feature_set: Literal["flat", "flat_spatial"] = "flat_spatial",
    scale: bool = True,
) -> Q1SplitData:
    """Apply the match-level split, fit preprocessor on train, transform all splits.

    ``feature_set='flat'`` is the baseline (used by the tree). ``'flat_spatial'``
    adds the spatial possession aggregates (used by the NN).
    """
    if feature_set == "flat":
        feature_names: tuple[str, ...] = Q1_FLAT_FEATURES
    elif feature_set == "flat_spatial":
        feature_names = Q1_FLAT_FEATURES + Q1_SPATIAL_FEATURES
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in df: {missing}")

    classes = tuple(sorted(df["label"].unique().tolist()))
    label_to_idx = {c: i for i, c in enumerate(classes)}

    def _slice(match_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        sub = df.loc[df["match_id"].astype(int).isin(set(match_ids))]
        X = sub[list(feature_names)].to_numpy(dtype=float)
        y = sub["label"].map(label_to_idx).to_numpy(dtype=np.int64)
        keys = sub[["match_id", "possession"]].reset_index(drop=True)
        return X, y, keys

    X_tr, y_tr, k_tr = _slice(splits.train)
    X_va, y_va, k_va = _slice(splits.val)
    X_te, y_te, k_te = _slice(splits.test)

    preprocessor = _build_preprocessor(scale=scale)
    X_tr_t = preprocessor.fit_transform(X_tr)
    X_va_t = preprocessor.transform(X_va) if len(X_va) else X_va.reshape(0, X_tr_t.shape[1])
    X_te_t = preprocessor.transform(X_te) if len(X_te) else X_te.reshape(0, X_tr_t.shape[1])

    return Q1SplitData(
        X_train=X_tr_t,
        y_train=y_tr,
        X_val=X_va_t,
        y_val=y_va,
        X_test=X_te_t,
        y_test=y_te,
        feature_names=feature_names,
        classes=classes,
        preprocessor=preprocessor,
        keys_train=k_tr,
        keys_val=k_va,
        keys_test=k_te,
    )


def class_weights_from_labels(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Inverse-frequency class weights normalized to mean 1.

    Mirrors sklearn's ``class_weight='balanced'`` heuristic so tree and NN agree.
    """
    counts = np.bincount(y, minlength=n_classes).astype(float)
    counts = np.where(counts == 0, 1.0, counts)
    weights = len(y) / (n_classes * counts)
    return weights


# ---------------------------------------------------------------------------
# Q2 — per-event, binary
# ---------------------------------------------------------------------------


@dataclass
class Q2SplitData:
    """Per-split feature matrix and binary label vector for Q2."""

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    preprocessor: Pipeline
    keys_train: pd.DataFrame  # match_id, event_uuid
    keys_val: pd.DataFrame
    keys_test: pd.DataFrame


def _expand_zone_one_hot(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``zone_<value>`` 0/1 columns for every value in ``Q2_ZONE_VALUES``.

    Any zone outside the known set falls into ``zone_unknown`` so the matrix
    stays a fixed width across runs.
    """
    out = df.copy()
    zone = out["zone"].astype(str).where(out["zone"].isin(Q2_ZONE_VALUES), other="unknown")
    for value in Q2_ZONE_VALUES:
        out[f"zone_{value}"] = (zone == value).astype(np.int8)
    return out


def load_q2_table(
    processed_dir: str | Path = DEFAULT_PROCESSED_DIR,
    *,
    include_spatial: bool = True,
) -> pd.DataFrame:
    """Return the joined Q2 table: labels + flat per-event (+ spatial per-event)."""
    processed = Path(processed_dir)
    labels = pd.read_parquet(processed / "labels_q2_shot_in_n.parquet")
    flat = pd.read_parquet(processed / "features_flat_event.parquet")

    df = labels.merge(
        flat,
        on=["match_id", "event_uuid"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_flat"),
    )
    df = _expand_zone_one_hot(df)

    if include_spatial:
        spatial = pd.read_parquet(processed / "features_spatial_event.parquet")
        df = df.merge(
            spatial,
            on=["match_id", "event_uuid"],
            how="left",
            validate="one_to_one",
        )
    return df


def prepare_q2_split(
    df: pd.DataFrame,
    splits: MatchSplits,
    *,
    feature_set: Literal["flat", "flat_spatial"] = "flat_spatial",
    scale: bool = True,
) -> Q2SplitData:
    """Apply the match-level split to a Q2 table; fit preprocessor on train only.

    ``feature_set='flat'`` is the logreg baseline. ``'flat_spatial'`` adds the
    six spatial freeze-frame features (used by the NN).
    """
    if feature_set == "flat":
        feature_names: tuple[str, ...] = Q2_FLAT_FEATURES
    elif feature_set == "flat_spatial":
        feature_names = Q2_FLAT_FEATURES + Q2_SPATIAL_FEATURES
    else:
        raise ValueError(f"Unknown feature_set: {feature_set!r}")

    missing = [c for c in feature_names if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in df: {missing}")

    def _slice(match_ids: Sequence[int]) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        sub = df.loc[df["match_id"].astype(int).isin(set(match_ids))]
        X = sub[list(feature_names)].to_numpy(dtype=float)
        y = sub["label"].astype(np.int64).to_numpy()
        keys = sub[["match_id", "event_uuid"]].reset_index(drop=True)
        return X, y, keys

    X_tr, y_tr, k_tr = _slice(splits.train)
    X_va, y_va, k_va = _slice(splits.val)
    X_te, y_te, k_te = _slice(splits.test)

    preprocessor = _build_preprocessor(scale=scale)
    X_tr_t = preprocessor.fit_transform(X_tr)
    X_va_t = preprocessor.transform(X_va) if len(X_va) else X_va.reshape(0, X_tr_t.shape[1])
    X_te_t = preprocessor.transform(X_te) if len(X_te) else X_te.reshape(0, X_tr_t.shape[1])

    return Q2SplitData(
        X_train=X_tr_t,
        y_train=y_tr,
        X_val=X_va_t,
        y_val=y_va,
        X_test=X_te_t,
        y_test=y_te,
        feature_names=feature_names,
        preprocessor=preprocessor,
        keys_train=k_tr,
        keys_val=k_va,
        keys_test=k_te,
    )


def pos_weight_from_labels(y: np.ndarray) -> float:
    """Return ``n_negative / n_positive`` for use as ``BCEWithLogitsLoss(pos_weight=...)``.

    With 5–15% positives this is the standard counter-balance: a positive
    misclassified contributes ``pos_weight`` times as much loss as a negative.
    Falls back to 1.0 when the slice has no positives or no negatives.
    """
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    if pos == 0 or neg == 0:
        return 1.0
    return neg / pos
