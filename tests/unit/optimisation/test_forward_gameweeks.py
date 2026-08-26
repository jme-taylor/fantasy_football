"""Which gameweeks the optimiser is allowed to plan."""

import polars as pl
import pytest

from fantasy_football.optimisation.inputs import forward_gameweeks
from fantasy_football.storage import database
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    PLAYER_WEEK,
    POINTS_PREDICTION,
)

SEASON = "2026-27"


def _prediction(gw: int, kind: str = FORWARD_KIND) -> dict:
    return {
        "season": SEASON,
        "gw": gw,
        "element": 1,
        "opponent": 10,
        "position": "MID",
        "predicted_points": 4.0,
        "model_version": "1",
        "prediction_kind": kind,
    }


def _played(gw: int) -> dict:
    return {
        "season": SEASON,
        "gw": gw,
        "element": 1,
        "name": "A Player",
        "position": "MID",
        "team": "Arsenal",
        "bonus": 0,
        "minutes": 90,
        "round": gw,
        "total_points": 5,
        "value": 50,
    }


@pytest.fixture
def seed(tmp_path, monkeypatch):
    """Seed forward predictions and played gameweeks into a tmp database."""

    def _seed(predictions: list[dict], played: list[dict]) -> None:
        db_path = tmp_path / "forward.duckdb"
        monkeypatch.setattr(database, "DATABASE_PATH", db_path)
        connection = get_connection(db_path)
        try:
            if predictions:
                POINTS_PREDICTION.upsert_current(
                    connection, pl.DataFrame(predictions), SEASON
                )
            if played:
                PLAYER_WEEK.upsert_current(
                    connection, pl.DataFrame(played), SEASON
                )
        finally:
            connection.close()

    return _seed


def test_forward_gameweeks_returns_unplayed_weeks_in_order(seed) -> None:
    """Pre-season, every forward gameweek is plannable."""
    seed([_prediction(gw) for gw in (3, 1, 2)], [])

    assert forward_gameweeks(SEASON) == [1, 2, 3]


def test_forward_gameweeks_drops_a_gameweek_already_played(seed) -> None:
    """A played week's forecast is frozen, not plannable.

    ``replace_partition`` deliberately keeps forward rows below the
    rewrite bound so the pre-deadline forecast survives. Planning on one
    would set start_gw to a gameweek that has already happened.
    """
    seed([_prediction(gw) for gw in (1, 2, 3)], [_played(1)])

    assert forward_gameweeks(SEASON) == [2, 3]


def test_forward_gameweeks_drops_every_played_gameweek(seed) -> None:
    """Mid-season the frozen tail can be many weeks long."""
    seed(
        [_prediction(gw) for gw in range(1, 8)],
        [_played(gw) for gw in (1, 2, 3)],
    )

    assert forward_gameweeks(SEASON) == [4, 5, 6, 7]


def test_forward_gameweeks_ignores_backfill_rows(seed) -> None:
    """Backfill is the in-sample rewrite, never a forecast."""
    seed(
        [_prediction(4), _prediction(5, kind=BACKFILL_KIND)],
        [],
    )

    assert forward_gameweeks(SEASON) == [4]


def test_forward_gameweeks_is_empty_when_nothing_is_predicted(seed) -> None:
    """No forecast at all reads as a finished season upstream."""
    seed([], [_played(1)])

    assert forward_gameweeks(SEASON) == []
