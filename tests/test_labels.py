import pandas as pd

from beyond_the_ball.features.labels import (
    label_q1_possession_terminal,
    label_q2_shot_in_n,
)


def _make_event(match_id, index, possession, team, etype, x, y, event_id=None):
    return {
        "match_id": match_id,
        "index": index,
        "possession": possession,
        "team": team,
        "type": etype,
        "location": [float(x), float(y)],
        "id": event_id or f"m{match_id}-i{index}",
    }


def test_q1_labels_shot_final_third_turnover_and_min_events():
    events = pd.DataFrame(
        [
            _make_event(1, 1, 1, "A", "Pass", 50, 40),
            _make_event(1, 2, 1, "A", "Carry", 70, 40),
            _make_event(1, 3, 1, "A", "Shot", 110, 40),
            _make_event(1, 4, 2, "A", "Pass", 60, 40),
            _make_event(1, 5, 2, "A", "Carry", 85, 40),
            _make_event(1, 6, 2, "A", "Pass", 90, 40),
            _make_event(1, 7, 3, "A", "Pass", 30, 40),
            _make_event(1, 8, 3, "A", "Carry", 50, 40),
            _make_event(1, 9, 3, "A", "Miscontrol", 55, 40),
            _make_event(1, 10, 4, "A", "Pass", 30, 40),
            _make_event(1, 11, 4, "A", "Shot", 100, 40),
        ]
    )

    out = label_q1_possession_terminal(events, min_events=3)

    by_poss = dict(zip(out["possession"], out["label"]))
    assert by_poss[1] == "shot"
    assert by_poss[2] == "final_third"
    assert by_poss[3] == "turnover"
    assert 4 not in by_poss


def test_q2_shot_in_n_basic_window_and_team_filter():
    events = pd.DataFrame(
        [
            _make_event(1, 1, 1, "A", "Pass", 50, 40, "e1"),
            _make_event(1, 2, 1, "A", "Carry", 60, 40, "e2"),
            _make_event(1, 3, 1, "A", "Shot", 110, 40, "e3"),
            _make_event(1, 4, 2, "B", "Pass", 30, 40, "e4"),
            _make_event(1, 5, 2, "B", "Pass", 35, 40, "e5"),
        ]
    )

    out = label_q2_shot_in_n(events, n=2, restrict_to_possession=True)
    by_event = dict(zip(out["event_uuid"], out["label"]))

    assert by_event["e1"] == 1
    assert by_event["e2"] == 1
    assert by_event["e3"] == 0
    assert by_event["e4"] == 0
    assert by_event["e5"] == 0


def test_q2_window_horizon_and_possession_boundary():
    events = pd.DataFrame(
        [
            _make_event(1, 1, 1, "A", "Pass", 50, 40, "p1"),
            _make_event(1, 2, 1, "A", "Pass", 52, 40, "p2"),
            _make_event(1, 3, 1, "A", "Pass", 54, 40, "p3"),
            _make_event(1, 4, 1, "A", "Pass", 56, 40, "p4"),
            _make_event(1, 5, 2, "A", "Shot", 110, 40, "shot"),
        ]
    )

    out_restricted = label_q2_shot_in_n(events, n=10, restrict_to_possession=True)
    by_event = dict(zip(out_restricted["event_uuid"], out_restricted["label"]))
    assert by_event["p1"] == 0

    out_unrestricted = label_q2_shot_in_n(events, n=10, restrict_to_possession=False)
    by_event_u = dict(zip(out_unrestricted["event_uuid"], out_unrestricted["label"]))
    assert by_event_u["p1"] == 1


def test_q2_horizon_n_excludes_distant_shots():
    events = pd.DataFrame(
        [_make_event(1, i + 1, 1, "A", "Pass", 50, 40, f"p{i}") for i in range(5)]
        + [_make_event(1, 6, 1, "A", "Shot", 110, 40, "shot")]
    )

    out = label_q2_shot_in_n(events, n=2, restrict_to_possession=True)
    by_event = dict(zip(out["event_uuid"], out["label"]))

    assert by_event["p0"] == 0
    assert by_event["p2"] == 0
    assert by_event["p3"] == 1
    assert by_event["p4"] == 1
