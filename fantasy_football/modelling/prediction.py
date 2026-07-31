import logging

import polars as pl

from fantasy_football.constants import (
    AWAY_FACTOR,
    DATA_FOLDER,
    HOME_FACTOR,
    OPPONENT_FACTOR_EXPONENT,
    ROLLING_WINDOW,
)
from fantasy_football.features.transformation import (
    KNOWN_POSITIONS,
    fill_missing_values_by_position,
    rolling_column_name,
)
from fantasy_football.modelling.models import MODELS_BY_POSITION

logger = logging.getLogger(__name__)

TRANSFORMED_DATA_FOLDER = DATA_FOLDER.joinpath("transformed")


def _latest_rolling_by_code(
    rolling: pl.DataFrame, current_season: str, as_of_gw: int | None
) -> pl.DataFrame:
    """Return each player's most recent rolling value, across all seasons.

    Looking across seasons rather than within the current one is what makes a
    pre-season prediction possible: before a ball is kicked the most recent
    row a player has is last season's final gameweek. Mid-season this is a
    no-op, because the most recent row is a current-season row.

    Identity is ``player_code`` rather than ``name`` or ``element``, since
    ``element`` is only unique within a season and display names collide.

    Parameters
    ----------
    rolling : pl.DataFrame
        The rolling points frame, carrying ``player_code`` and the rolling
        column.
    current_season : str
        The season being predicted.
    as_of_gw : int | None
        When set, current-season rows after this gameweek are excluded, so a
        backtest cannot see its own future. Prior seasons are always in
        scope. When None, every row up to and including the current season
        is eligible.

    Returns
    -------
    pl.DataFrame
        One row per ``player_code``, with a ``baseline`` column.

    Notes
    -----
    Eligibility is always ordering-aware, not just equality-aware: a row from
    any season later than ``current_season`` is excluded regardless of
    ``as_of_gw``, since season strings sort chronologically
    ("2025-26" < "2026-27") and a season that hasn't been reached yet can
    never be a legitimate source for a baseline. This keeps the ``as_of_gw is
    None`` path and the ``as_of_gw`` cutoff path consistent with each other:
    both exclude the future, the cutoff just narrows it further within the
    current season.
    """
    rolling_col = rolling_column_name("total_points", ROLLING_WINDOW)
    eligible = rolling.filter(
        pl.col("player_code").is_not_null()
        & (pl.col("season") <= current_season)
    )
    if as_of_gw is not None:
        eligible = eligible.filter(
            (pl.col("season") != current_season) | (pl.col("gw") <= as_of_gw)
        )
    if eligible.is_empty():
        return pl.DataFrame(
            schema={"player_code": pl.Int64, "baseline": pl.Float64}
        )
    return (
        eligible.sort(["season", "gw"])
        .group_by("player_code")
        .agg(pl.col(rolling_col).last().alias("baseline"))
    )


def _baselines(
    rolling: pl.DataFrame,
    current_season: str,
    as_of_gw: int | None = None,
    roster: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return one row per player: rolling baseline plus roster fields.

    Parameters
    ----------
    rolling: pl.DataFrame
        The rolling points DataFrame.
    current_season: str
        The current season.
    as_of_gw : int | None, optional
        When set, restrict the baseline to each player's latest rolling row at
        or before this gameweek (leak-free for a past-window backtest). When
        None, the latest row is used. Defaults to None.
    roster : pl.DataFrame | None, optional
        Who is in the league right now, with ``name``, ``position``, ``team``,
        ``element`` and ``player_code``. When given, identity comes from here
        and only form comes from ``rolling`` -- which is what lets a season
        with no played gameweeks produce baselines at all. When None or empty,
        identity falls back to the current season's rolling rows. Defaults to
        None.

    Returns
    -------
    pl.DataFrame
        The baselines DataFrame, with columns ``name``, ``position``,
        ``team``, ``element`` and ``baseline``.
    """
    rolling_col = rolling_column_name("total_points", ROLLING_WINDOW)
    if roster is not None and not roster.is_empty():
        latest = _latest_rolling_by_code(rolling, current_season, as_of_gw)
        joined = roster.join(latest, on="player_code", how="left")
        cold = sorted(
            joined.filter(pl.col("baseline").is_null())["name"].to_list()
        )
        if cold:
            logger.info(
                "%d players have no prior rolling baseline and take the "
                "positional average: %s",
                len(cold),
                cold,
            )
        filled = fill_missing_values_by_position(joined, "baseline")
        unfilled = sorted(
            filled.filter(pl.col("baseline").is_null())["name"].to_list()
        )
        if unfilled:
            logger.warning(
                "Dropping %d players with no baseline and no positional "
                "average to fall back on: %s",
                len(unfilled),
                unfilled,
            )
            filled = filled.filter(pl.col("baseline").is_not_null())
        return filled.select(
            "name", "position", "team", "element", "baseline"
        )

    current = rolling.filter(pl.col("season") == current_season)
    if as_of_gw is not None:
        current = current.filter(pl.col("gw") <= as_of_gw)
    if current.is_empty():
        return current.select(
            "name",
            "position",
            "team",
            "element",
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
        "element",
        pl.col(rolling_col).alias("baseline"),
    ).unique(subset=["name"], keep="first")


def _elo_as_of(
    team_elo: pl.DataFrame,
    fixtures: pl.DataFrame,
    team_col: str,
    out_col: str,
) -> pl.DataFrame:
    """Attach as-of ELO for ``team_col`` into ``fixtures`` as ``out_col``.

    Parameters
    ----------
    team_elo: pl.DataFrame
        The team ELO DataFrame.
    fixtures: pl.DataFrame
        The fixtures DataFrame.
    team_col: str
        The column name to use for the team.
    out_col: str
        The column name to use for the output.

    Returns
    -------
    pl.DataFrame
        The joined DataFrame.
    """
    intervals = team_elo.rename({"team": team_col, "elo": out_col})
    joined = fixtures.sort("kickoff_date").join_asof(
        intervals.sort("from_date"),
        left_on="kickoff_date",
        right_on="from_date",
        by=team_col,
        strategy="backward",
    )
    return joined.drop("to_date", "from_date")


def _apply_models(fx: pl.DataFrame, baselines: pl.DataFrame) -> pl.DataFrame:
    """Join baselines, build factors, and apply the per-position model.

    Parameters
    ----------
    fx : pl.DataFrame
        Fixtures enriched with player and opponent ELO.
    baselines : pl.DataFrame
        One row per player with their latest rolling baseline and team.

    Returns
    -------
    pl.DataFrame
        Per-(player, future_gw) rows with a ``predicted_points`` column.
    """
    joined = fx.join(baselines, on="team", how="inner").with_columns(
        (pl.col("player_team_elo") / pl.col("opponent_team_elo"))
        .pow(OPPONENT_FACTOR_EXPONENT)
        .alias("opponent_factor"),
        pl.when(pl.col("is_home"))
        .then(HOME_FACTOR)
        .otherwise(AWAY_FACTOR)
        .alias("home_away_factor"),
    )

    unknown = set(joined["position"].unique().to_list()) - set(KNOWN_POSITIONS)
    for position in sorted(unknown):
        logger.warning(
            "No model for position %r; dropping its predictions", position
        )

    scored_frames = []
    for position, model in MODELS_BY_POSITION.items():
        sub = joined.filter(pl.col("position") == position)
        if sub.is_empty():
            continue
        scored_frames.append(
            sub.with_columns(model.predict(sub).alias("predicted_points"))
        )

    if not scored_frames:
        scored = joined.with_columns(
            pl.lit(None, dtype=pl.Float64).alias("predicted_points")
        )
    else:
        scored = pl.concat(scored_frames, how="vertical")

    return scored.select(
        "name",
        pl.col("element").alias("player_id"),
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


def _predict(
    current_season: str,
    horizon_n: int | None = None,
    as_of_gw: int | None = None,
) -> pl.DataFrame:
    """Produce per-(player, future_gw) point predictions and return them.

    Loads the three input tables: rolling points, fixtures, and team ELO, then
    gets each player's baseline (latest current rolling points) and figures
    out which gameweeks require prediction. If `horizon_n` is not provided, it
    will predict on all future gameweeks. It then attaches ELO ratings for
    each fixture and calculates the predicted points for each player.

    Parameters
    ----------
    current_season: str
        The current season.
    horizon_n: int | None
        The number of future gameweeks to predict. If None, all future gameweeks
        will be predicted.
    as_of_gw : int | None
        Pivot gameweek. When set, predictions cover gameweeks after as_of_gw
        and baselines use only form at or before it (leak-free past-window
        backtest). When None, the pivot is the latest completed current-season
        gameweek. Defaults to None.

    Returns
    -------
    pl.DataFrame
        The predictions DataFrame.
    """
    rolling = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("rolling_points.csv"),
        try_parse_dates=True,
    )
    baselines = _baselines(rolling, current_season, as_of_gw)
    fixtures = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("fixtures_enriched.csv"),
        try_parse_dates=True,
    ).filter(pl.col("season") == current_season)
    team_elo = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("team_elo.csv"),
        try_parse_dates=True,
    )

    last_completed = (
        as_of_gw
        if as_of_gw is not None
        else (
            rolling.filter(pl.col("season") == current_season)["gw"].max() or 0
        )
    )
    if horizon_n is None:
        max_future_gw = fixtures.filter(pl.col("gw") > last_completed)[
            "gw"
        ].max()
        horizon_n = (
            (max_future_gw - last_completed)
            if max_future_gw is not None
            else 0
        )
    # gameweeks to predict on
    horizon_gws = list(
        range(last_completed + 1, last_completed + 1 + horizon_n)
    )
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

    fx = _apply_models(fx, baselines)
    return fx


def predict_points(
    current_season: str,
    horizon_n: int | None = None,
    as_of_gw: int | None = None,
) -> pl.DataFrame:
    """Compute predictions, write them to predictions.csv, and return them.

    Thin IO wrapper around :func:`_predict`. See that function for the
    prediction logic and parameter meanings.

    Parameters
    ----------
    current_season : str
        The current season.
    horizon_n : int | None
        Number of future gameweeks to predict. None predicts all of them.
    as_of_gw : int | None
        Pivot gameweek for a leak-free past-window backtest.

    Returns
    -------
    pl.DataFrame
        The predictions DataFrame.
    """
    predictions = _predict(current_season, horizon_n, as_of_gw)
    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    predictions.write_csv(TRANSFORMED_DATA_FOLDER.joinpath("predictions.csv"))
    logger.info(
        "Wrote %d predictions covering gws %s",
        predictions.height,
        sorted(set(predictions["gw"].to_list())),
    )
    return predictions
