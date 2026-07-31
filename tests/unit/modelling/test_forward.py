"""Tests for building forward-scoring frames from a snapshot."""

from datetime import datetime

import polars as pl
import pytest

from fantasy_football.modelling.forward import (
    build_forward_fixtures,
    forward_player_weeks,
    last_played_gw,
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
