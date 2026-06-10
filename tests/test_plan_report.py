from fantasy_football.fpl_types import PlayerGameweekExpectedPoints
from fantasy_football.plan_report import _ordered_rows


def _p(name: str, pts: float) -> PlayerGameweekExpectedPoints:
    return PlayerGameweekExpectedPoints(
        player_id=hash(name) % 1000, player_name=name, expected_points=pts
    )


def test_ordered_rows_groups_by_position_then_points_desc() -> None:
    """Rows come out GK, DEF, MID, FWD; within a group, highest xPts first."""
    players = [
        _p("Striker", 5.0),
        _p("Keeper", 4.0),
        _p("Mid1", 9.0),
        _p("Mid2", 2.0),
    ]
    positions = {
        "Striker": "FWD",
        "Keeper": "GK",
        "Mid1": "MID",
        "Mid2": "MID",
    }

    rows = _ordered_rows(players, positions)

    assert [(pos, p.player_name) for pos, p in rows] == [
        ("GK", "Keeper"),
        ("MID", "Mid1"),
        ("MID", "Mid2"),
        ("FWD", "Striker"),
    ]


def test_ordered_rows_missing_position_renders_as_question_mark() -> None:
    """A name absent from the positions map gets a '?' position and sorts last."""
    players = [_p("Known", 5.0), _p("Unknown", 9.0)]
    positions = {"Known": "MID"}

    rows = _ordered_rows(players, positions)

    assert [(pos, p.player_name) for pos, p in rows] == [
        ("MID", "Known"),
        ("?", "Unknown"),
    ]
