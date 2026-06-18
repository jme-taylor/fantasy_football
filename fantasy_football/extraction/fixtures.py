"""Build the ``team_fixture`` table from true fixture lists.

Historic seasons come from Vaastav's per-season ``fixtures.csv``; the current
live season comes from the FPL ``/api/fixtures/`` endpoint. Both sources are
FPL-shaped (gameweek, ISO kickoff string, home/away team ids), so a single
transform explodes each fixture into two team-perspective rows. See
``docs/superpowers/specs/2026-06-17-team-fixture-table-design.md``.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import polars as pl
import requests

from fantasy_football.extraction.extractor import DataExtractor
from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.extraction.seasons import DataSource, source_for_season
from fantasy_football.storage.database import (
    TEAM_FIXTURE_SCHEMA,
    fixture_seasons_present,
    seasons_present,
    upsert_current_fixtures,
    write_immutable_fixtures,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# (gameweek, ISO kickoff string, home team id, away team id)
NormalisedFixture = tuple[int, str, int, int]


def _parse_kickoff(kickoff: str) -> datetime:
    """Parse an ISO kickoff string to a naive UTC datetime.

    Parameters
    ----------
    kickoff : str
        ISO 8601 timestamp, e.g. ``"2023-08-11T19:00:00Z"``.

    Returns
    -------
    datetime
        The instant in UTC with the timezone stripped, so it maps to a duckdb
        ``TIMESTAMP`` (tz-naive) column.
    """
    parsed = datetime.fromisoformat(kickoff.replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc).replace(tzinfo=None)


def fixtures_to_team_rows(
    fixtures: list[NormalisedFixture],
    teams_by_id: dict[int, str],
    season: str,
) -> pl.DataFrame:
    """Explode FPL-shaped fixtures into two team-perspective rows each.

    Parameters
    ----------
    fixtures : list[NormalisedFixture]
        ``(gw, kickoff_iso, home_team_id, away_team_id)`` tuples.
    teams_by_id : dict[int, str]
        Maps a team id (as used in ``fixtures``) to its display name.
    season : str
        Short-form season string, e.g. ``"2023-24"``.

    Returns
    -------
    pl.DataFrame
        One row per team per fixture, columns and dtypes per
        ``TEAM_FIXTURE_SCHEMA``. Double gameweeks naturally produce two rows
        for the affected team.

    Raises
    ------
    KeyError
        If a fixture references a team id absent from ``teams_by_id``.
    """
    rows: list[dict] = []
    for gw, kickoff_iso, home_id, away_id in fixtures:
        kickoff = _parse_kickoff(kickoff_iso)
        home = teams_by_id[home_id]
        away = teams_by_id[away_id]
        rows.append(
            {
                "season": season,
                "gw": gw,
                "team": home,
                "is_home": True,
                "opposition": away,
                "kickoff_time": kickoff,
            }
        )
        rows.append(
            {
                "season": season,
                "gw": gw,
                "team": away,
                "is_home": False,
                "opposition": home,
                "kickoff_time": kickoff,
            }
        )
    if not rows:
        return pl.DataFrame(schema=TEAM_FIXTURE_SCHEMA)
    return pl.DataFrame(rows).cast(TEAM_FIXTURE_SCHEMA, strict=False)


def build_current_fixtures(season: str, api: FplAPI) -> pl.DataFrame:
    """Build ``team_fixture`` rows for the current season from the FPL API.

    Parameters
    ----------
    season : str
        Short-form season string for the live season, e.g. ``"2025-26"``.
    api : FplAPI
        The FPL API client. ``get_fixtures`` already drops fixtures with no
        gameweek or kickoff time.

    Returns
    -------
    pl.DataFrame
        Team-fixture rows in the canonical schema.
    """
    teams_by_id = {team.id: team.name for team in api.get_teams()}
    fixtures: list[NormalisedFixture] = [
        (fixture.event, fixture.kickoff_time, fixture.team_h, fixture.team_a)
        for fixture in api.get_fixtures().fixtures
    ]
    return fixtures_to_team_rows(fixtures, teams_by_id, season)


def build_vaastav_fixtures(
    season: str, extractor: DataExtractor
) -> pl.DataFrame:
    """Build ``team_fixture`` rows for a historic season from Vaastav.

    Reads the season's ``fixtures.csv`` and ``teams.csv`` from the Vaastav
    repo. Fixtures with a missing gameweek or kickoff time (postponed /
    unscheduled) are dropped. Team ids are season-local and resolved via that
    season's ``teams.csv``.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2023-24"``.
    extractor : DataExtractor
        Provides ``_read_csv`` to download a repo CSV into a Polars frame.

    Returns
    -------
    pl.DataFrame
        Team-fixture rows in the canonical schema.
    """
    teams_df = extractor._read_csv(f"data/{season}/teams.csv")
    teams_by_id = dict(
        zip(teams_df["id"].to_list(), teams_df["name"].to_list())
    )
    fixtures_df = extractor._read_csv(f"data/{season}/fixtures.csv")
    scheduled = fixtures_df.filter(
        pl.col("event").is_not_null()
        & pl.col("kickoff_time").is_not_null()
        & (pl.col("kickoff_time") != "")
    )
    fixtures: list[NormalisedFixture] = [
        (
            int(row["event"]),
            row["kickoff_time"],
            int(row["team_h"]),
            int(row["team_a"]),
        )
        for row in scheduled.iter_rows(named=True)
    ]
    return fixtures_to_team_rows(fixtures, teams_by_id, season)


def load_fixtures(
    connection: "DuckDBPyConnection",
    current_season: str,
    *,
    api: FplAPI | None = None,
    extractor: DataExtractor | None = None,
) -> None:
    """Populate ``team_fixture`` for the current season and historic seasons.

    Two distinct behaviours run on every call:

    * **Historic seasons** — driven by the seasons present in ``player_week``.
      Each season already in ``team_fixture`` is skipped (immutable, already
      complete). For remaining seasons, Vaastav is the only supported historic
      source; a non-current FCI-era season (beyond Vaastav's last season but
      not the live season) has no kickoff source and is logged and skipped.  A
      Vaastav HTTP error for a single season is also logged and skipped rather
      than aborting the entire run.

    * **Current season** — the live season is **always** re-fetched from the
      FPL API and upserted on every call, regardless of whether it already
      appears in ``player_week`` or ``team_fixture``.  This ensures the fixture
      schedule (kickoff times, new gameweeks) is available for prediction and
      optimisation even before player-week stats are published, and stays
      up-to-date as the season progresses.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Open connection to the database.
    current_season : str
        The current live season, e.g. ``"2025-26"``.
    api : FplAPI | None, optional
        FPL API client for the current season. Defaults to a new ``FplAPI``.
    extractor : DataExtractor | None, optional
        Vaastav CSV reader. Defaults to a new ``DataExtractor``.
    """
    api = api or FplAPI()
    extractor = extractor or DataExtractor()
    already = fixture_seasons_present(connection)

    # Season strings sort lexicographically in calendar order (e.g. "2023-24" < "2024-25").
    for season in sorted(seasons_present(connection)):
        if season in already or season == current_season:
            continue
        if source_for_season(season) != DataSource.VAASTAV:
            logger.warning(
                "No fixture source for non-current FCI season %s; skipping.",
                season,
            )
            continue
        try:
            frame = build_vaastav_fixtures(season, extractor)
        except requests.HTTPError as exc:
            logger.warning(
                "Could not fetch Vaastav fixtures for %s (%s); skipping.",
                season,
                exc,
            )
            continue
        write_immutable_fixtures(connection, frame, season)

    # Always refresh the current season (kickoff times and new gameweeks can
    # change), matching the player-week upsert behaviour.
    frame = build_current_fixtures(current_season, api)
    upsert_current_fixtures(connection, frame, current_season)
