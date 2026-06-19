"""Unit tests for fixture extraction adapters and transform."""

from datetime import datetime
from unittest.mock import MagicMock

import polars as pl
import pytest
import requests

from fantasy_football.extraction.fixtures import (
    build_current_fixtures,
    fixtures_to_team_rows,
    load_fixtures,
)
from fantasy_football.extraction.seasons import (
    DataSource,
    source_for_season,
)
from fantasy_football.fpl_types import (
    FplFixture,
    FplFixtures,
    FplTeamInfo,
)
from fantasy_football.storage.database import (
    fixture_seasons_present,
    get_connection,
    load_team_fixture,
)


def test_transform_emits_two_rows_per_fixture() -> None:
    """Each fixture becomes a home row and an away row."""
    teams = {1: "Arsenal", 2: "Chelsea"}
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    result = fixtures_to_team_rows(fixtures, teams, season="2023-24")
    assert result.height == 2
    home = result.filter(pl.col("team") == "Arsenal").row(0, named=True)
    away = result.filter(pl.col("team") == "Chelsea").row(0, named=True)
    assert home["opposition"] == "Chelsea"
    assert home["is_home"] is True
    assert away["opposition"] == "Arsenal"
    assert away["is_home"] is False
    for row in (home, away):
        assert row["season"] == "2023-24"
        assert row["gw"] == 1
        assert row["kickoff_time"] == datetime(2023, 8, 11, 19, 0)


def test_transform_kickoff_is_naive_utc() -> None:
    """kickoff_time is stored as a naive UTC datetime (no timezone)."""
    teams = {1: "Arsenal", 2: "Chelsea"}
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    result = fixtures_to_team_rows(fixtures, teams, season="2023-24")
    assert result.schema["kickoff_time"] == pl.Datetime("us")
    assert result["kickoff_time"][0].tzinfo is None


def test_transform_handles_double_gameweek() -> None:
    """A team with two fixtures in one gw yields two rows."""
    teams = {1: "Arsenal", 2: "Chelsea", 3: "Spurs"}
    fixtures = [
        (29, "2026-03-14T15:00:00Z", 1, 2),
        (29, "2026-03-17T19:45:00Z", 1, 3),
    ]
    result = fixtures_to_team_rows(fixtures, teams, season="2025-26")
    arsenal = result.filter(pl.col("team") == "Arsenal").sort("kickoff_time")
    assert arsenal.height == 2
    assert arsenal["opposition"].to_list() == ["Chelsea", "Spurs"]


def test_transform_empty_returns_typed_empty_frame() -> None:
    """No fixtures yields an empty frame with the canonical schema."""
    result = fixtures_to_team_rows([], {1: "Arsenal"}, season="2023-24")
    assert result.is_empty()
    assert result.columns == [
        "season",
        "gw",
        "team",
        "is_home",
        "opposition",
        "kickoff_time",
    ]


def test_transform_unknown_team_id_raises() -> None:
    """An unknown team id raises KeyError."""
    teams = {1: "Arsenal"}  # id 2 missing
    fixtures = [(1, "2023-08-11T19:00:00Z", 1, 2)]
    with pytest.raises(KeyError):
        fixtures_to_team_rows(fixtures, teams, season="2023-24")


def _team(id: int, name: str) -> FplTeamInfo:
    return FplTeamInfo(
        id=id, code=id * 10, name=name, short_name=name[:3].upper()
    )


def _fix(event: int, team_h: int, team_a: int, kickoff: str) -> FplFixture:
    return FplFixture(
        code=event * 100 + team_h,
        event=event,
        finished=False,
        id=event * 100 + team_h,
        kickoff_time=kickoff,
        team_a=team_a,
        team_a_difficulty=3,
        team_h=team_h,
        team_h_difficulty=3,
    )


def _api(teams, fixtures) -> MagicMock:
    api = MagicMock()
    api.get_teams.return_value = teams
    api.get_fixtures.return_value = FplFixtures(fixtures=fixtures)
    return api


def test_build_current_fixtures_from_api() -> None:
    """build_current_fixtures maps FPL fixtures into team_fixture rows."""
    teams = [_team(1, "Arsenal"), _team(2, "Chelsea")]
    fixtures = [_fix(1, 1, 2, "2025-08-16T15:00:00Z")]
    result = build_current_fixtures("2025-26", _api(teams, fixtures))
    assert result.height == 2
    home = result.filter(pl.col("is_home")).row(0, named=True)
    assert home["team"] == "Arsenal"
    assert home["opposition"] == "Chelsea"
    assert home["gw"] == 1
    assert home["season"] == "2025-26"


def test_build_current_fixtures_empty_when_no_fixtures() -> None:
    """No fixtures yields an empty canonical frame."""
    result = build_current_fixtures("2025-26", _api([_team(1, "Arsenal")], []))
    assert result.is_empty()
    assert "kickoff_time" in result.columns


from fantasy_football.extraction.fixtures import (
    build_vaastav_fixtures,  # noqa: E402
)


def _vaastav_extractor(fixtures_df, teams_df) -> MagicMock:
    """DataExtractor stub whose _read_csv routes by path suffix."""
    extractor = MagicMock()

    def _read_csv(path: str):
        if path.endswith("teams.csv"):
            return teams_df
        if path.endswith("fixtures.csv"):
            return fixtures_df
        raise AssertionError(f"unexpected path {path}")

    extractor._read_csv.side_effect = _read_csv
    return extractor


def test_build_vaastav_fixtures_maps_local_team_ids() -> None:
    """Season-local team ids are resolved via teams.csv."""
    teams_df = pl.DataFrame({"id": [6, 13], "name": ["Arsenal", "Liverpool"]})
    fixtures_df = pl.DataFrame(
        {
            "event": [1],
            "kickoff_time": ["2023-08-11T19:00:00Z"],
            "team_h": [6],
            "team_a": [13],
        }
    )
    extractor = _vaastav_extractor(fixtures_df, teams_df)
    result = build_vaastav_fixtures("2023-24", extractor)
    assert result.height == 2
    home = result.filter(pl.col("is_home")).row(0, named=True)
    assert home["team"] == "Arsenal"
    assert home["opposition"] == "Liverpool"
    assert home["season"] == "2023-24"
    extractor._read_csv.assert_any_call("data/2023-24/teams.csv")
    extractor._read_csv.assert_any_call("data/2023-24/fixtures.csv")


def test_build_vaastav_fixtures_drops_unscheduled_rows() -> None:
    """Rows with a null event or blank kickoff_time are dropped."""
    teams_df = pl.DataFrame({"id": [6, 13], "name": ["Arsenal", "Liverpool"]})
    fixtures_df = pl.DataFrame(
        {
            "event": [1, None],
            "kickoff_time": ["2023-08-11T19:00:00Z", None],
            "team_h": [6, 6],
            "team_a": [13, 13],
        }
    )
    extractor = _vaastav_extractor(fixtures_df, teams_df)
    result = build_vaastav_fixtures("2023-24", extractor)
    assert result.height == 2  # only the one scheduled fixture, exploded
    assert result["gw"].unique().to_list() == [1]


def _seed_player_week(conn, seasons) -> None:
    for i, season in enumerate(seasons):
        conn.execute(
            "INSERT INTO player_week (season, gw, element) VALUES (?, ?, ?)",
            [season, 1, i + 1],
        )


def test_load_fixtures_routes_historic_and_current(
    tmp_path, monkeypatch
) -> None:
    """Historic seasons use Vaastav; the current season uses the API."""
    conn = get_connection(tmp_path / "t.duckdb")
    try:
        _seed_player_week(conn, ["2023-24", "2025-26"])

        def fake_vaastav(season, extractor):
            return fixtures_to_team_rows(
                [(1, "2023-08-11T19:00:00Z", 1, 2)],
                {1: "Arsenal", 2: "Chelsea"},
                season,
            )

        def fake_current(season, api):
            return fixtures_to_team_rows(
                [(1, "2025-08-16T15:00:00Z", 1, 2)],
                {1: "Arsenal", 2: "Chelsea"},
                season,
            )

        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_vaastav_fixtures",
            fake_vaastav,
        )
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_current_fixtures",
            fake_current,
        )

        load_fixtures(conn, "2025-26", api=MagicMock(), extractor=MagicMock())

        assert fixture_seasons_present(conn) == {"2023-24", "2025-26"}
        assert load_team_fixture(conn).height == 4
    finally:
        conn.close()


def test_load_fixtures_skips_seasons_already_present(
    tmp_path, monkeypatch
) -> None:
    """A season already in team_fixture is not rebuilt."""
    conn = get_connection(tmp_path / "t.duckdb")
    try:
        _seed_player_week(conn, ["2023-24"])
        calls = []
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_vaastav_fixtures",
            lambda season, extractor: (
                calls.append(season)
                or fixtures_to_team_rows(
                    [(1, "2023-08-11T19:00:00Z", 1, 2)],
                    {1: "A", 2: "B"},
                    season,
                )
            ),
        )
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_current_fixtures",
            lambda season, api: fixtures_to_team_rows([], {}, season),
        )
        load_fixtures(conn, "2099-00", api=MagicMock(), extractor=MagicMock())
        load_fixtures(conn, "2099-00", api=MagicMock(), extractor=MagicMock())
        assert calls == ["2023-24"]  # built once, skipped on the second run
    finally:
        conn.close()


def test_load_fixtures_vaastav_fetch_failure_is_skipped(
    tmp_path, monkeypatch
) -> None:
    """A Vaastav fetch error logs and skips rather than aborting the run."""
    conn = get_connection(tmp_path / "t.duckdb")
    try:
        _seed_player_week(conn, ["2016-17", "2023-24"])

        def flaky(season, extractor):
            if season == "2016-17":
                raise requests.HTTPError("404")
            return fixtures_to_team_rows(
                [(1, "2023-08-11T19:00:00Z", 1, 2)],
                {1: "A", 2: "B"},
                season,
            )

        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_vaastav_fixtures",
            flaky,
        )
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_current_fixtures",
            lambda season, api: fixtures_to_team_rows([], {}, season),
        )
        load_fixtures(conn, "2099-00", api=MagicMock(), extractor=MagicMock())
        assert fixture_seasons_present(conn) == {"2023-24"}
    finally:
        conn.close()


def test_load_fixtures_skips_non_current_fci_season(
    tmp_path, monkeypatch
) -> None:
    """A non-current FCI-era season in player_week is warned about and skipped.

    "2030-31" is beyond Vaastav's last season (2024-25) so it routes to FCI,
    but it is not the current season ("2099-00"), so there is no fixture source
    for it and it must be skipped.  "2023-24" is a Vaastav season and should
    be loaded normally.  The current season "2099-00" is always upserted but
    the empty frame contributes nothing.
    """
    # Confirm routing up-front so the test is self-documenting.
    assert source_for_season("2030-31") == DataSource.FCI

    conn = get_connection(tmp_path / "t.duckdb")
    try:
        _seed_player_week(conn, ["2023-24", "2030-31"])

        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_vaastav_fixtures",
            lambda season, extractor: fixtures_to_team_rows(
                [(1, "2023-08-11T19:00:00Z", 1, 2)],
                {1: "Arsenal", 2: "Chelsea"},
                season,
            ),
        )
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_current_fixtures",
            lambda season, api: fixtures_to_team_rows([], {}, season),
        )

        load_fixtures(conn, "2099-00", api=MagicMock(), extractor=MagicMock())

        # "2030-31" was skipped; "2099-00" is empty; only "2023-24" landed.
        assert fixture_seasons_present(conn) == {"2023-24"}
    finally:
        conn.close()
