import pandas as pd
import pytest

from beyond_the_ball.data.splits import (
    MatchSplits,
    assign_split_column,
    load_splits,
    save_splits,
    slice_by_split,
    split_matches,
)


def _flatten(splits: MatchSplits) -> set[int]:
    return set(splits.train) | set(splits.val) | set(splits.test)


def test_split_is_a_partition_with_no_overlap():
    match_ids = list(range(100))
    splits = split_matches(match_ids, seed=0)

    assert _flatten(splits) == set(match_ids)
    assert set(splits.train).isdisjoint(splits.val)
    assert set(splits.train).isdisjoint(splits.test)
    assert set(splits.val).isdisjoint(splits.test)


def test_split_ratios_match_request():
    match_ids = list(range(1000))
    splits = split_matches(match_ids, val_size=0.15, test_size=0.15, seed=1)
    assert len(splits.test) == 150
    assert len(splits.val) == 150
    assert len(splits.train) == 700


def test_split_is_deterministic_given_seed():
    a = split_matches(range(50), seed=7)
    b = split_matches(range(50), seed=7)
    c = split_matches(range(50), seed=8)
    assert a == b
    assert a != c


def test_stratified_split_keeps_each_group_in_every_split():
    # Two groups: 100 matches each. Each split should contain matches from both.
    match_ids = list(range(200))
    strata = {m: ("A" if m < 100 else "B") for m in match_ids}
    splits = split_matches(match_ids, seed=3, strata=strata)

    for name in ("train", "val", "test"):
        ids = getattr(splits, name)
        groups = {strata[m] for m in ids}
        assert groups == {"A", "B"}, f"{name} missing a group: {groups}"

    # Per-group ratio holds approximately.
    a_test = sum(1 for m in splits.test if strata[m] == "A")
    b_test = sum(1 for m in splits.test if strata[m] == "B")
    assert a_test == 15
    assert b_test == 15


def test_stratified_split_raises_for_unmapped_match():
    with pytest.raises(ValueError):
        split_matches([1, 2, 3], strata={1: "A", 2: "B"})


def test_invalid_sizes_rejected():
    with pytest.raises(ValueError):
        split_matches([1, 2, 3], val_size=0.6, test_size=0.5)
    with pytest.raises(ValueError):
        split_matches([1, 2, 3], val_size=0.0)


def test_assign_split_column_and_slice():
    splits = MatchSplits(train=(1, 2), val=(3,), test=(4,))
    df = pd.DataFrame({"match_id": [1, 1, 2, 3, 4, 4], "v": range(6)})

    out = assign_split_column(df, splits)
    assert out.loc[out["match_id"] == 1, "split"].unique().tolist() == ["train"]
    assert out.loc[out["match_id"] == 3, "split"].unique().tolist() == ["val"]
    assert out.loc[out["match_id"] == 4, "split"].unique().tolist() == ["test"]

    val_rows = slice_by_split(df, splits, "val")
    assert val_rows["match_id"].unique().tolist() == [3]


def test_assign_split_raises_on_unknown_match():
    splits = MatchSplits(train=(1,), val=(2,), test=(3,))
    df = pd.DataFrame({"match_id": [1, 99]})
    with pytest.raises(ValueError):
        assign_split_column(df, splits)


def test_save_and_load_splits_roundtrip(tmp_path):
    splits = split_matches(range(30), seed=11)
    path = tmp_path / "splits.parquet"
    save_splits(splits, path)
    loaded = load_splits(path)
    assert loaded == splits
