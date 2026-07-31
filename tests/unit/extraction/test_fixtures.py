"""Unit tests for fixture extraction adapters and transform."""

from datetime import datetime
from typing import TYPE_CHECKING
from unittest.mock import MagicMock

import duckdb
import polars as pl
import pytest
import requests

from fantasy_football.extraction.fixtures import (
    build_current_fixtures,
    fixtures_to_team_rows,
    load_fixtures,
    season_from_kickoffs,
    stored_season_is_valid,
)
from fantasy_football.fpl_types import (
    FplFixture,
    FplFixtures,
    FplTeamInfo,
)
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import TEAM_FIXTURE

if TYPE_CHECKING:
    import pytest_mock


def _fixture_rows(season: str, kickoff: datetime) -> pl.DataFrame:
    """Build one minimal team_fixture row for a season.

    Parameters
    ----------
    season : str
        Short-form season label to store the row under.
    kickoff : datetime
        The kickoff instant for the row.

    Returns
    -------
    pl.DataFrame
        A single team-fixture row in the canonical schema.
    """
    return pl.DataFrame(
        {
            "season": [season],
            "gw": [1],
            "team": ["Arsenal"],
            "is_home": [True],
            "opposition": ["Chelsea"],
            "kickoff_time": [kickoff],
        }
    ).cast(TEAM_FIXTURE.schema, strict=False)


def test_stored_season_is_valid_accepts_matching_kickoffs(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A season whose kickoffs fall inside its own window is valid."""
    TEAM_FIXTURE.upsert_current(
        db, _fixture_rows("2025-26", datetime(2025, 8, 15, 19, 0)), "2025-26"
    )
    assert stored_season_is_valid(db, "2025-26") is True


def test_stored_season_is_valid_rejects_next_seasons_payload(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """The live 2025-26 corruption: 2026-27 kickoffs under a 2025-26 label."""
    TEAM_FIXTURE.upsert_current(
        db, _fixture_rows("2025-26", datetime(2026, 8, 21, 19, 0)), "2025-26"
    )
    assert stored_season_is_valid(db, "2025-26") is False


def test_stored_season_is_valid_treats_absent_season_as_invalid(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A season with no rows needs fetching, so it is not valid."""
    assert stored_season_is_valid(db, "2025-26") is False


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

        assert TEAM_FIXTURE.seasons_present(conn) == {"2023-24", "2025-26"}
        assert TEAM_FIXTURE.load(conn).height == 4
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
        assert TEAM_FIXTURE.seasons_present(conn) == {"2023-24"}
    finally:
        conn.close()


def test_load_fixtures_repairs_mislabelled_stored_season(
    tmp_path, monkeypatch
) -> None:
    """A stored season with the wrong payload is re-derived, not skipped.

    "2025-26" is present in ``team_fixture`` but holds kickoffs that
    actually belong to "2026-27" (the live corruption this task fixes).
    Presence alone must not be treated as proof of correctness: Vaastav is
    queried again and the corrected rows overwrite the bad ones.
    """
    conn = get_connection(tmp_path / "t.duckdb")
    try:
        _seed_player_week(conn, ["2025-26"])
        TEAM_FIXTURE.upsert_current(
            conn,
            fixtures_to_team_rows(
                [(1, "2026-08-21T19:00:00Z", 1, 2)],
                {1: "Arsenal", 2: "Chelsea"},
                "2025-26",
            ),
            "2025-26",
        )

        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_vaastav_fixtures",
            lambda season, extractor: fixtures_to_team_rows(
                [(1, "2025-08-15T19:00:00Z", 1, 2)],
                {1: "Arsenal", 2: "Chelsea"},
                season,
            ),
        )
        monkeypatch.setattr(
            "fantasy_football.extraction.fixtures.build_current_fixtures",
            lambda season, api: fixtures_to_team_rows([], {}, season),
        )

        load_fixtures(conn, "2099-00", api=MagicMock(), extractor=MagicMock())

        repaired = TEAM_FIXTURE.load(conn).filter(
            pl.col("season") == "2025-26"
        )
        assert repaired["kickoff_time"].unique().to_list() == [
            datetime(2025, 8, 15, 19, 0)
        ]
    finally:
        conn.close()


def test_season_from_kickoffs_uses_august_boundary() -> None:
    """A season runs August to May, so both months map to one label."""
    kickoffs = [
        datetime(2025, 8, 15, 19, 0),
        datetime(2026, 5, 24, 15, 0),
    ]
    assert season_from_kickoffs(kickoffs) == "2025-26"


def test_season_from_kickoffs_handles_july_as_prior_season() -> None:
    """A July kickoff belongs to the season that started the year before."""
    assert season_from_kickoffs([datetime(2026, 7, 5, 15, 0)]) == "2025-26"


def test_build_current_fixtures_rejects_mismatched_season(
    mocker: "pytest_mock.MockerFixture",
) -> None:
    """Writing next season's payload under this season's label must fail."""

    class _Team:
        def __init__(self, team_id: int, name: str) -> None:
            self.id = team_id
            self.name = name

    class _Fixture:
        event = 1
        kickoff_time = "2026-08-21T19:00:00Z"
        team_h = 1
        team_a = 2

    api = mocker.Mock()
    api.get_teams.return_value = [_Team(1, "Arsenal"), _Team(2, "Chelsea")]
    api.get_fixtures.return_value = mocker.Mock(fixtures=[_Fixture()])

    with pytest.raises(ValueError, match="2026-27"):
        build_current_fixtures("2025-26", api)
