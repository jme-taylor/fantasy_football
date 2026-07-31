import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import polars as pl
import requests

from fantasy_football.extraction.extractor import DataExtractor
from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.storage.tables import PLAYER_WEEK, TEAM_FIXTURE

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
        ``TEAM_FIXTURE.schema``. Double gameweeks naturally produce two rows
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
        return pl.DataFrame(schema=TEAM_FIXTURE.schema)
    return pl.DataFrame(rows).cast(TEAM_FIXTURE.schema, strict=False)


def season_from_kickoffs(kickoffs: list[datetime]) -> str:
    """Infer the short-form season a set of kickoff times belongs to.

    A Premier League season runs August to May. Anything from August
    onwards belongs to the season starting that calendar year; anything
    from January to July belongs to the season that started the year
    before.

    Parameters
    ----------
    kickoffs : list[datetime]
        Naive UTC kickoff instants. Must not be empty.

    Returns
    -------
    str
        Short-form season string, e.g. ``"2025-26"``.

    Raises
    ------
    ValueError
        If ``kickoffs`` is empty.
    """
    if not kickoffs:
        raise ValueError("Cannot infer a season from no kickoff times.")
    earliest = min(kickoffs)
    start_year = earliest.year if earliest.month >= 8 else earliest.year - 1
    return f"{start_year}-{str(start_year + 1)[2:]}"


def stored_season_is_valid(
    connection: "DuckDBPyConnection", season: str
) -> bool:
    """Report whether a stored season's kickoff times match its label.

    Presence alone is not proof of correctness: a season fetched from the
    live API while ``CURRENT_SEASON`` was stale carries the *next*
    season's fixtures under this season's label. Comparing the earliest
    stored kickoff against the label catches that.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    season : str
        Short-form season string to check.

    Returns
    -------
    bool
        True when the season has rows and they belong to it. False when
        the season is absent or mislabelled.
    """
    stored = TEAM_FIXTURE.load(connection).filter(pl.col("season") == season)
    if stored.is_empty():
        return False
    kickoffs = stored["kickoff_time"].drop_nulls().to_list()
    if not kickoffs:
        return False
    return season_from_kickoffs(kickoffs) == season


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

    Raises
    ------
    ValueError
        If the fixture payload's inferred season contradicts the label
        passed in, indicating ``CURRENT_SEASON`` is stale.
    """
    teams_by_id = {team.id: team.name for team in api.get_teams()}
    fixtures: list[NormalisedFixture] = [
        (fixture.event, fixture.kickoff_time, fixture.team_h, fixture.team_a)
        for fixture in api.get_fixtures().fixtures
    ]
    kickoffs = [_parse_kickoff(kickoff) for _, kickoff, _, _ in fixtures]
    if kickoffs:
        actual = season_from_kickoffs(kickoffs)
        if actual != season:
            raise ValueError(
                f"FPL returned {actual} fixtures but they were about to be "
                f"stored as {season}. CURRENT_SEASON is probably stale."
            )
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
      A season is re-derived from Vaastav unless ``stored_season_is_valid``
      confirms its stored kickoff times already match its label; presence
      alone is not proof of correctness, since a season fetched while
      ``CURRENT_SEASON`` was stale can carry the next season's fixtures
      under this season's label. Vaastav still publishes ``fixtures.csv``
      and ``teams.csv`` for every season it has, including ones beyond its
      last season of *player* data; a season Vaastav genuinely lacks 404s
      and is logged and skipped rather than aborting the entire run.

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

    # Season strings sort lexicographically in calendar order (e.g. "2023-24" < "2024-25").
    for season in sorted(PLAYER_WEEK.seasons_present(connection)):
        if season == current_season:
            continue
        if stored_season_is_valid(connection, season):
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
        TEAM_FIXTURE.upsert_current(connection, frame, season)

    # Always refresh the current season (kickoff times and new gameweeks can
    # change), matching the player-week upsert behaviour.
    frame = build_current_fixtures(current_season, api)
    TEAM_FIXTURE.upsert_current(connection, frame, current_season)
