"""Match-level train/val/test splits.

All events from one match go to the same split — event-level random splits
leak because consecutive events are correlated. Stratification by an optional
group key (e.g. competition) keeps the per-group share roughly constant in
each split.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SPLIT_NAMES: tuple[str, str, str] = ("train", "val", "test")
DEFAULT_VAL_SIZE: float = 0.15
DEFAULT_TEST_SIZE: float = 0.15
DEFAULT_SEED: int = 42


@dataclass(frozen=True)
class MatchSplits:
    """Match IDs assigned to each split."""

    train: tuple[int, ...]
    val: tuple[int, ...]
    test: tuple[int, ...]

    def as_dict(self) -> dict[str, tuple[int, ...]]:
        return {"train": self.train, "val": self.val, "test": self.test}

    def assignment(self) -> dict[int, str]:
        out: dict[int, str] = {}
        for name in SPLIT_NAMES:
            for mid in getattr(self, name):
                out[int(mid)] = name
        return out


def _split_one_group(
    match_ids: np.ndarray,
    val_size: float,
    test_size: float,
    rng: np.random.Generator,
) -> tuple[list[int], list[int], list[int]]:
    """Split a single group of match IDs, allocating test then val then train."""
    ids = np.array(sorted(int(m) for m in match_ids))
    n = len(ids)
    if n == 0:
        return [], [], []

    perm = rng.permutation(n)
    shuffled = ids[perm]

    n_test = int(round(n * test_size))
    n_val = int(round(n * val_size))

    # Guarantee at least one match in each split when the group has >=3 matches,
    # otherwise prefer to leave val/test empty rather than starve train.
    if n >= 3:
        n_test = max(1, n_test)
        n_val = max(1, n_val)
    if n_val + n_test >= n:
        n_test = min(n_test, max(0, n - 1))
        n_val = min(n_val, max(0, n - 1 - n_test))

    test = shuffled[:n_test].tolist()
    val = shuffled[n_test : n_test + n_val].tolist()
    train = shuffled[n_test + n_val :].tolist()
    return train, val, test


def split_matches(
    match_ids: Iterable[int],
    *,
    val_size: float = DEFAULT_VAL_SIZE,
    test_size: float = DEFAULT_TEST_SIZE,
    seed: int = DEFAULT_SEED,
    strata: Mapping[int, object] | pd.Series | None = None,
) -> MatchSplits:
    """Partition match IDs into train/val/test, optionally stratified by group.

    The split is deterministic given ``seed``. If ``strata`` is provided each
    distinct group is split independently, which keeps the share of each group
    in every split close to its overall share.
    """
    if not 0.0 < val_size < 1.0 or not 0.0 < test_size < 1.0:
        raise ValueError("val_size and test_size must be in (0, 1).")
    if val_size + test_size >= 1.0:
        raise ValueError("val_size + test_size must be < 1.")

    unique_ids = sorted({int(m) for m in match_ids})
    if not unique_ids:
        return MatchSplits((), (), ())

    rng = np.random.default_rng(seed)

    if strata is None:
        groups = {0: np.array(unique_ids)}
    else:
        if isinstance(strata, pd.Series):
            mapping: dict[int, object] = strata.to_dict()
        else:
            mapping = dict(strata)
        unknown = [m for m in unique_ids if m not in mapping]
        if unknown:
            raise ValueError(
                f"Stratum missing for {len(unknown)} match_ids (e.g. {unknown[:3]})."
            )
        groups_dict: dict[object, list[int]] = {}
        for m in unique_ids:
            groups_dict.setdefault(mapping[m], []).append(m)
        # Sort group keys to make iteration deterministic regardless of insertion order.
        groups = {
            k: np.array(groups_dict[k]) for k in sorted(groups_dict, key=lambda x: str(x))
        }

    train: list[int] = []
    val: list[int] = []
    test: list[int] = []
    for _, ids in groups.items():
        tr, va, te = _split_one_group(ids, val_size, test_size, rng)
        train.extend(tr)
        val.extend(va)
        test.extend(te)

    return MatchSplits(
        train=tuple(sorted(train)),
        val=tuple(sorted(val)),
        test=tuple(sorted(test)),
    )


def assign_split_column(
    df: pd.DataFrame,
    splits: MatchSplits,
    *,
    match_id_col: str = "match_id",
    out_col: str = "split",
) -> pd.DataFrame:
    """Return a copy of ``df`` with a ``split`` column derived from match_id."""
    if match_id_col not in df.columns:
        raise ValueError(f"DataFrame missing column: {match_id_col}")
    assignment = splits.assignment()
    out = df.copy()
    out[out_col] = out[match_id_col].astype(int).map(assignment)
    if out[out_col].isna().any():
        missing = out.loc[out[out_col].isna(), match_id_col].unique()[:5].tolist()
        raise ValueError(f"match_ids not in splits (e.g. {missing}).")
    return out


def slice_by_split(
    df: pd.DataFrame,
    splits: MatchSplits,
    split: str,
    *,
    match_id_col: str = "match_id",
) -> pd.DataFrame:
    """Return rows of ``df`` whose match_id belongs to ``split``."""
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}.")
    ids = set(getattr(splits, split))
    return df.loc[df[match_id_col].astype(int).isin(ids)].reset_index(drop=True)


def save_splits(splits: MatchSplits, path: str | Path) -> Path:
    """Persist splits as a tidy parquet of (match_id, split)."""
    rows = [(mid, name) for name, ids in splits.as_dict().items() for mid in ids]
    out = pd.DataFrame(rows, columns=["match_id", "split"]).sort_values("match_id")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(target, index=False)
    return target


def load_splits(path: str | Path) -> MatchSplits:
    """Load splits previously written by :func:`save_splits`."""
    df = pd.read_parquet(Path(path))
    by_split = {name: tuple(sorted(int(m) for m in df.loc[df["split"] == name, "match_id"]))
                for name in SPLIT_NAMES}
    return MatchSplits(**by_split)
