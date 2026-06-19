import logging
from datetime import datetime

import polars as pl

from fantasy_football.constants import DATA_FOLDER
from fantasy_football.extraction.fpl import FplAPI

logger = logging.getLogger(__name__)

TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")


def enrich_fixtures(api: FplAPI, season: str) -> pl.DataFrame:
    """Return one row per (team, season, gw) for the given season.

    Each fixture produces two rows: one for the home team and one for the
    away team. Double gameweeks (a team playing twice in one event) produce
    two rows for that team-gw.

    Parameters
    ----------
    api: FplAPI
        The FPL API client.
    season: str
        The season to build the fixtures for.

    Returns
    -------
    pl.DataFrame
        The enriched fixtures DataFrame.
    """
    teams = {team.id: team.name for team in api.get_teams()}
    rows: list[dict] = []
    for fixture in api.get_fixtures().fixtures:
        kickoff_date = datetime.fromisoformat(
            fixture.kickoff_time.replace("Z", "+00:00")
        ).date()
        home_name = teams[fixture.team_h]
        away_name = teams[fixture.team_a]
        rows.append(
            {
                "team": home_name,
                "opponent_team": away_name,
                "is_home": True,
                "kickoff_date": kickoff_date,
                "season": season,
                "gw": fixture.event,
            }
        )
        rows.append(
            {
                "team": away_name,
                "opponent_team": home_name,
                "is_home": False,
                "kickoff_date": kickoff_date,
                "season": season,
                "gw": fixture.event,
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "team": pl.Utf8,
                "opponent_team": pl.Utf8,
                "is_home": pl.Boolean,
                "kickoff_date": pl.Date,
                "season": pl.Utf8,
                "gw": pl.Int64,
            }
        )
    return pl.DataFrame(rows)


def build_fixtures_enriched(season: str) -> pl.DataFrame:
    """Build the enriched fixtures table and write it to CSV.

    Parameters
    ----------
    season: str
        The season to build the fixtures for.

    Returns
    -------
    pl.DataFrame
        The enriched fixtures DataFrame.
    """
    df = enrich_fixtures(FplAPI(), season)
    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    df.write_csv(TRANSFORMED_DATA_FOLDER.joinpath("fixtures_enriched.csv"))
    logger.info("Wrote %d enriched fixture rows for %s", df.height, season)
    return df


def count_fixtures_in_gw(team_fixtures: pl.DataFrame) -> pl.DataFrame:
    """Count how many fixtures each team has in each gameweek.

    Groups ``team_fixture`` rows by ``(season, gw, team)`` and counts them.
    Because ``team_fixture`` is keyed on ``(season, gw, team, opposition)``,
    each row is a distinct fixture: a normal week counts 1, a double gameweek
    counts 2. Blank gameweeks have no rows and are absent from the output, so a
    missing team-gw should be read as 0 fixtures by the caller.

    ``season`` is part of the grouping because ``gw`` repeats across seasons;
    omitting it would pool different seasons' gameweeks together.

    Parameters
    ----------
    team_fixtures : pl.DataFrame
        The ``team_fixture`` table, with at least ``season``, ``gw`` and
        ``team`` columns.

    Returns
    -------
    pl.DataFrame
        Columns ``season``, ``gw``, ``team`` and ``fixtures_in_gw`` (Int64),
        sorted by ``(season, gw, team)``.

    """
    return (
        team_fixtures.group_by(["season", "gw", "team"])
        .agg(pl.len().alias("fixtures_in_gw"))
        .with_columns(pl.col("fixtures_in_gw").cast(pl.Int64))
        .sort(["season", "gw", "team"])
    )
