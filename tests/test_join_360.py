import pandas as pd

from beyond_the_ball.data.join_360 import filter_open_play, join_events_with_360


def test_join_events_with_360_joins_by_event_uuid():
    events = pd.DataFrame(
        {
            "id": ["evt-1", "evt-2", "evt-3"],
            "match_id": [11, 11, 11],
            "type": ["Pass", "Shot", "Pass"],
            "play_pattern": ["Regular Play", "From Corner", "Regular Play"],
        }
    )
    frames = pd.DataFrame(
        {
            "event_uuid": ["evt-1", "evt-3"],
            "freeze_frame": [[{"teammate": True}], [{"teammate": False}]],
            "visible_area": [[0, 0, 1, 1], [0, 0, 1, 1]],
        }
    )

    joined = join_events_with_360(events, frames)

    assert len(joined) == 3
    assert joined["event_uuid"].tolist() == ["evt-1", "evt-2", "evt-3"]
    assert joined.loc[joined["id"] == "evt-2", "freeze_frame"].isna().all()
    assert joined.loc[joined["id"] == "evt-1", "n_visible_players"].iloc[0] == 1


def test_filter_open_play_removes_set_pieces():
    canonical = pd.DataFrame(
        {
            "id": ["1", "2", "3", "4"],
            "play_pattern": [
                "Regular Play",
                "From Corner",
                "Regular Play",
                "From Throw In",
            ],
            "type": ["Pass", "Pass", "Free Kick", "Pass"],
        }
    )

    filtered = filter_open_play(canonical)

    assert filtered["id"].tolist() == ["1"]

