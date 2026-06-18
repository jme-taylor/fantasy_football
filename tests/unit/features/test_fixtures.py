from datetime import date
from unittest.mock import MagicMock

import polars as pl
import pytest

from fantasy_football.features.fixtures import (
    count_fixtures_in_gw,
    enrich_fixtures,
)
from fantasy_football.fpl_types import FplFixture, FplFixtures, FplTeamInfo


def _team(id: int, name: str) -> FplTeamInfo:
    return FplTeamInfo(
        id=id, code=id * 10, name=name, short_name=name[:3].upper()
    )


def _fix(
    event: int, team_h: int, team_a: int, kickoff: str, id: int = 0
) -> FplFixture:
    return FplFixture(
        code=id or event * 100 + team_h,
        event=event,
        finished=False,
        id=id or event * 100 + team_h,
        kickoff_time=kickoff,
        team_a=team_a,
        team_a_difficulty=3,
        team_h=team_h,
        team_h_difficulty=3,
    )


def _api(teams: list[FplTeamInfo], fixtures: list[FplFixture]) -> MagicMock:
    api = MagicMock()
    api.get_teams.return_value = teams
    api.get_fixtures.return_value = FplFixtures(fixtures=fixtures)
    return api


def test_enrich_fixtures_emits_one_row_per_team_per_fixture() -> None:
    """Verify enrich_fixtures emits one row per team per fixture."""
    teams = [_team(1, "Arsenal"), _team(2, "Chelsea")]
    fixtures = [
        _fix(event=1, team_h=1, team_a=2, kickoff="2025-08-16T15:00:00Z")
    ]

    result = enrich_fixtures(_api(teams, fixtures), season="2025-26")

    assert result.height == 2
    home = result.filter(pl.col("team") == "Arsenal").row(0, named=True)
    away = result.filter(pl.col("team") == "Chelsea").row(0, named=True)
    assert home["opponent_team"] == "Chelsea"
    assert home["is_home"] is True
    assert away["opponent_team"] == "Arsenal"
    assert away["is_home"] is False
    for row in (home, away):
        assert row["season"] == "2025-26"
        assert row["gw"] == 1
        assert row["kickoff_date"] == date(2025, 8, 16)


def test_enrich_fixtures_handles_double_gameweek() -> None:
    """Verify enrich_fixtures correctly handles double gameweeks."""
    teams = [_team(1, "Arsenal"), _team(2, "Chelsea"), _team(3, "Spurs")]
    fixtures = [
        _fix(
            event=29, team_h=1, team_a=2, kickoff="2026-03-14T15:00:00Z", id=1
        ),
        _fix(
            event=29, team_h=1, team_a=3, kickoff="2026-03-17T19:45:00Z", id=2
        ),
    ]

    result = enrich_fixtures(_api(teams, fixtures), season="2025-26")

    arsenal = result.filter(pl.col("team") == "Arsenal").sort("kickoff_date")
    assert arsenal.height == 2
    assert arsenal["opponent_team"].to_list() == ["Chelsea", "Spurs"]


def test_enrich_fixtures_skips_fixtures_with_no_event() -> None:
    """Verify enrich_fixtures returns empty frame for fixtures with no event."""
    teams = [_team(1, "Arsenal"), _team(2, "Chelsea")]
    fixtures = []  # parse_fixtures already drops event-less fixtures
    result = enrich_fixtures(_api(teams, fixtures), season="2025-26")
    assert result.is_empty()


def test_enrich_fixtures_unknown_team_id_raises() -> None:
    """Verify enrich_fixtures raises KeyError for unknown team IDs."""
    teams = [_team(1, "Arsenal")]  # team id 2 not registered
    fixtures = [
        _fix(event=1, team_h=1, team_a=2, kickoff="2025-08-16T15:00:00Z")
    ]
    with pytest.raises(KeyError):
        enrich_fixtures(_api(teams, fixtures), season="2025-26")


@pytest.fixture
def sample_team_fixtures() -> pl.DataFrame:
    """team_fixture rows covering a normal week, a double gameweek and two seasons."""
    return pl.DataFrame(
        {
            "season": ["2023-24", "2023-24", "2023-24", "2023-24", "2024-25"],
            "gw": [1, 1, 2, 2, 1],
            "team": ["Arsenal", "Chelsea", "Arsenal", "Arsenal", "Arsenal"],
            "is_home": [True, False, True, False, True],
            "opposition": ["Chelsea", "Arsenal", "Spurs", "Everton", "Forest"],
            "kickoff_time": [None, None, None, None, None],
        }
    )


def test_count_fixtures_in_gw_normal_week(
    sample_team_fixtures: pl.DataFrame,
) -> None:
    """A team with one fixture in a gameweek counts 1."""
    result = count_fixtures_in_gw(sample_team_fixtures)
    row = result.filter(
        (pl.col("season") == "2023-24")
        & (pl.col("gw") == 1)
        & (pl.col("team") == "Arsenal")
    )
    assert row.get_column("fixtures_in_gw").item() == 1


def test_count_fixtures_in_gw_double_gameweek(
    sample_team_fixtures: pl.DataFrame,
) -> None:
    """Two fixtures in one gameweek (different oppositions) count 2."""
    result = count_fixtures_in_gw(sample_team_fixtures)
    row = result.filter(
        (pl.col("season") == "2023-24")
        & (pl.col("gw") == 2)
        & (pl.col("team") == "Arsenal")
    )
    assert row.get_column("fixtures_in_gw").item() == 2


def test_count_fixtures_in_gw_partitions_by_season(
    sample_team_fixtures: pl.DataFrame,
) -> None:
    """Same gw/team in another season is not pooled."""
    result = count_fixtures_in_gw(sample_team_fixtures)
    row = result.filter(
        (pl.col("season") == "2024-25")
        & (pl.col("gw") == 1)
        & (pl.col("team") == "Arsenal")
    )
    assert row.get_column("fixtures_in_gw").item() == 1


def test_count_fixtures_in_gw_counts_each_side_once(
    sample_team_fixtures: pl.DataFrame,
) -> None:
    """Home and away sides of one fixture each count 1 for their own team."""
    result = count_fixtures_in_gw(sample_team_fixtures)
    chelsea = result.filter(
        (pl.col("season") == "2023-24")
        & (pl.col("gw") == 1)
        & (pl.col("team") == "Chelsea")
    )
    assert chelsea.get_column("fixtures_in_gw").item() == 1


def test_count_fixtures_in_gw_omits_blanks(
    sample_team_fixtures: pl.DataFrame,
) -> None:
    """A team with no rows in a gameweek produces no output row (blank = absent)."""
    result = count_fixtures_in_gw(sample_team_fixtures)
    chelsea_gw2 = result.filter(
        (pl.col("season") == "2023-24")
        & (pl.col("gw") == 2)
        & (pl.col("team") == "Chelsea")
    )
    assert chelsea_gw2.height == 0
