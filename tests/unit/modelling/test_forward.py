"""Tests for building forward-scoring frames from a snapshot."""

from datetime import datetime

import polars as pl
import pytest

from fantasy_football.modelling.forward import (
    build_forward_fixtures,
    forward_player_weeks,
    last_played_gw,
    latest_snapshot,
)


@pytest.fixture
def snapshot() -> pl.DataFrame:
    """Two Arsenal players captured at one instant.

    Returns
    -------
    pl.DataFrame
        Snapshot rows for season 2026-27.
    """
    return pl.DataFrame(
        {
            "season": ["2026-27"] * 2,
            "captured_at": [datetime(2026, 7, 31, 12, 0)] * 2,
            "element": [1, 2],
            "value": [55, 90],
            "team": ["Arsenal", "Arsenal"],
            "position": ["DEF", "FWD"],
            "chance_of_playing_this_round": [100, 75],
        }
    )


@pytest.fixture
def fixtures() -> pl.DataFrame:
    """Arsenal fixtures: a single GW1 and a double GW2.

    Returns
    -------
    pl.DataFrame
        Team-fixture rows for season 2026-27.
    """
    return pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "gw": [1, 2, 2],
            "team": ["Arsenal"] * 3,
            "is_home": [True, False, True],
            "opposition": ["Chelsea", "Everton", "Fulham"],
            "kickoff_time": [
                datetime(2026, 8, 21, 19, 0),
                datetime(2026, 8, 28, 14, 0),
                datetime(2026, 8, 30, 14, 0),
            ],
        }
    )


def test_last_played_gw_is_zero_pre_season() -> None:
    """An empty player_week for the season means nothing has been played."""
    empty = pl.DataFrame({"season": ["2025-26"], "gw": [38], "element": [1]})
    assert last_played_gw(empty, "2026-27") == 0


def test_latest_snapshot_returns_only_the_latest_capture() -> None:
    """Only rows sharing the max captured_at for the season are kept."""
    snapshot = pl.DataFrame(
        {
            "season": ["2026-27"] * 3,
            "captured_at": [
                datetime(2026, 7, 30, 12, 0),
                datetime(2026, 7, 31, 12, 0),
                datetime(2026, 7, 31, 12, 0),
            ],
            "element": [1, 1, 2],
            "value": [50, 55, 90],
            "team": ["Arsenal", "Arsenal", "Arsenal"],
            "position": ["DEF", "DEF", "FWD"],
            "chance_of_playing_this_round": [100, 100, 75],
        }
    )

    result = latest_snapshot(snapshot, "2026-27")

    assert result.height == 2
    assert result["captured_at"].n_unique() == 1
    assert result["captured_at"][0] == datetime(2026, 7, 31, 12, 0)
    assert sorted(result["element"].to_list()) == [1, 2]


def test_latest_snapshot_empty_for_missing_season() -> None:
    """A season with no captures yields an empty frame, not an error."""
    snapshot = pl.DataFrame(
        {
            "season": ["2025-26"],
            "captured_at": [datetime(2025, 8, 1, 12, 0)],
            "element": [1],
            "value": [50],
            "team": ["Arsenal"],
            "position": ["DEF"],
            "chance_of_playing_this_round": [100],
        }
    )

    result = latest_snapshot(snapshot, "2026-27")

    assert result.is_empty()


def test_build_forward_fixtures_expands_snapshot_across_fixtures(
    snapshot: pl.DataFrame, fixtures: pl.DataFrame
) -> None:
    """Each player gets one row per club fixture, doubles included."""
    ids = {"Chelsea": 7, "Everton": 8, "Fulham": 9}

    result = build_forward_fixtures(
        snapshot, fixtures, "2026-27", from_gw=1, team_name_to_id=ids
    )

    # 2 players x 3 fixtures.
    assert result.height == 6
    assert result["minutes"].null_count() == 6
    assert result["minutes"].dtype == pl.Int64
    gw2 = result.filter((pl.col("gw") == 2) & (pl.col("element") == 1))
    assert sorted(gw2["opponent"].to_list()) == [8, 9]


def test_build_forward_fixtures_respects_the_gameweek_floor(
    snapshot: pl.DataFrame, fixtures: pl.DataFrame
) -> None:
    """Played gameweeks are excluded from the forward universe."""
    ids = {"Chelsea": 7, "Everton": 8, "Fulham": 9}

    result = build_forward_fixtures(
        snapshot, fixtures, "2026-27", from_gw=2, team_name_to_id=ids
    )

    assert set(result["gw"].to_list()) == {2}


def test_forward_player_weeks_collapses_double_gameweeks(
    snapshot: pl.DataFrame, fixtures: pl.DataFrame
) -> None:
    """The week grain has one row per player-gameweek, not per fixture."""
    ids = {"Chelsea": 7, "Everton": 8, "Fulham": 9}
    forward = build_forward_fixtures(
        snapshot, fixtures, "2026-27", from_gw=1, team_name_to_id=ids
    )

    weeks = forward_player_weeks(forward)

    # 2 players x 2 gameweeks.
    assert weeks.height == 4
    assert weeks.select(["season", "gw", "element"]).is_unique().all()


def test_forward_player_weeks_keeps_the_earliest_kickoff_leg() -> None:
    """On a double, the surviving row is the earlier-kickoff fixture.

    The two legs of the double share every column except ``value`` and
    ``kickoff_time``, so a wrong sort direction or ``keep`` argument
    would flip which value survives -- without changing row counts or
    uniqueness, which the other test already covers.
    """
    forward = pl.DataFrame(
        {
            "season": ["2026-27", "2026-27"],
            "gw": [2, 2],
            "element": [1, 1],
            "opponent": [8, 9],
            "kickoff_time": [
                datetime(2026, 8, 28, 14, 0),
                datetime(2026, 8, 30, 14, 0),
            ],
            "minutes": pl.Series([None, None], dtype=pl.Int64),
            "position": ["DEF", "DEF"],
            "team": ["Arsenal", "Arsenal"],
            "value": [55, 99],
            "chance_of_playing_this_round": [100, 100],
        }
    )

    weeks = forward_player_weeks(forward)

    assert weeks.height == 1
    assert weeks["value"][0] == 55
