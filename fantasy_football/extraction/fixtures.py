"""Build the ``team_fixture`` table from true fixture lists.

Historic seasons come from Vaastav's per-season ``fixtures.csv``; the current
live season comes from the FPL ``/api/fixtures/`` endpoint. Both sources are
FPL-shaped (gameweek, ISO kickoff string, home/away team ids), so a single
transform explodes each fixture into two team-perspective rows. See
``docs/superpowers/specs/2026-06-17-team-fixture-table-design.md``.
"""

import logging
from datetime import datetime, timezone

import polars as pl

from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.storage.database import TEAM_FIXTURE_SCHEMA

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
