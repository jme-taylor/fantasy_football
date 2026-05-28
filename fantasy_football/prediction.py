"""Apply the baseline heuristic to produce per-player point predictions."""

from __future__ import annotations

import logging

import polars as pl

from fantasy_football.constants import (
    AWAY_FACTOR,
    DATA_FOLDER,
    HOME_FACTOR,
    OPPONENT_FACTOR_EXPONENT,
    ROLLING_WINDOW,
)
from fantasy_football.data_transformation import rolling_column_name

logger = logging.getLogger(__name__)

TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")


def _baselines(rolling: pl.DataFrame, current_season: str) -> pl.DataFrame:
    """Return one row per player: latest current-season rolling value + team."""
    rolling_col = rolling_column_name("total_points", ROLLING_WINDOW)
    current = rolling.filter(pl.col("season") == current_season)
    if current.is_empty():
        return current.select(
            "name",
            "position",
            "team",
            pl.col(rolling_col).alias("baseline"),
        )
    dropped = sorted(
        current.filter(pl.col("team").is_null())["name"].unique().to_list()
    )
    for name in dropped:
        logger.debug("Dropping player %r — no team in current season", name)
    with_team = current.filter(pl.col("team").is_not_null())
    latest_gw = with_team.group_by("name").agg(pl.col("gw").max().alias("gw"))
    latest = with_team.join(latest_gw, on=["name", "gw"], how="inner")
    return latest.select(
        "name",
        "position",
        "team",
        pl.col(rolling_col).alias("baseline"),
    ).unique(subset=["name"], keep="first")


def _elo_as_of(
    team_elo: pl.DataFrame,
    fixtures: pl.DataFrame,
    team_col: str,
    out_col: str,
) -> pl.DataFrame:
    """Attach as-of ELO for ``team_col`` into ``fixtures`` as ``out_col``."""
    intervals = team_elo.rename({"team": team_col, "elo": out_col})
    joined = fixtures.sort("kickoff_date").join_asof(
        intervals.sort("from_date"),
        left_on="kickoff_date",
        right_on="from_date",
        by=team_col,
        strategy="backward",
    )
    return joined.drop("to_date", "from_date")


def predict_points(current_season: str, horizon_n: int) -> pl.DataFrame:
    """Produce per-(player, future_gw) point predictions and write CSV."""
    rolling = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("rolling_points.csv"),
        try_parse_dates=True,
    )
    baselines = _baselines(rolling, current_season)
    fixtures = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("fixtures_enriched.csv"),
        try_parse_dates=True,
    ).filter(pl.col("season") == current_season)
    team_elo = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("team_elo.csv"),
        try_parse_dates=True,
    )

    last_completed = (
        rolling.filter(pl.col("season") == current_season)["gw"].max() or 0
    )
    horizon_gws = list(range(last_completed + 1, last_completed + 1 + horizon_n))
    fixtures = fixtures.filter(pl.col("gw").is_in(horizon_gws))
    if fixtures.is_empty():
        logger.info("No fixtures found in horizon %s", horizon_gws)

    fx = _elo_as_of(team_elo, fixtures, "team", "player_team_elo")
    fx = _elo_as_of(team_elo, fx, "opponent_team", "opponent_team_elo")

    median_elo = team_elo["elo"].median()
    for col, team_col in (
        ("player_team_elo", "team"),
        ("opponent_team_elo", "opponent_team"),
    ):
        missing = fx.filter(pl.col(col).is_null())[team_col].unique().to_list()
        for name in missing:
            logger.warning(
                "Missing ELO for team %r — using median %.1f for %s",
                name,
                median_elo,
                col,
            )
        fx = fx.with_columns(pl.col(col).fill_null(median_elo))

    joined = (
        fx.join(baselines, on="team", how="inner")
        .with_columns(
            (pl.col("player_team_elo") / pl.col("opponent_team_elo"))
            .pow(OPPONENT_FACTOR_EXPONENT)
            .alias("opponent_factor"),
            pl.when(pl.col("is_home"))
            .then(HOME_FACTOR)
            .otherwise(AWAY_FACTOR)
            .alias("home_away_factor"),
        )
        .with_columns(
            (
                pl.col("baseline")
                * pl.col("opponent_factor")
                * pl.col("home_away_factor")
            ).alias("predicted_points")
        )
        .select(
            "name",
            "position",
            "team",
            "season",
            "gw",
            "opponent_team",
            "is_home",
            "baseline",
            "player_team_elo",
            "opponent_team_elo",
            "opponent_factor",
            "home_away_factor",
            "predicted_points",
        )
    )

    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    joined.write_csv(TRANSFORMED_DATA_FOLDER.joinpath("predictions.csv"))
    logger.info(
        "Wrote %d predictions covering gws %s",
        joined.height,
        sorted(set(joined["gw"].to_list())),
    )
    return joined
