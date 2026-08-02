import logging

import polars as pl

from fantasy_football.constants import (
    AWAY_FACTOR,
    DATA_FOLDER,
    HOME_FACTOR,
    OPPONENT_FACTOR_EXPONENT,
    ROLLING_WINDOW,
)
from fantasy_football.extraction.seasons import previous_season
from fantasy_football.features.roster import current_roster
from fantasy_football.features.transformation import (
    KNOWN_POSITIONS,
    fill_missing_values_by_position,
    rolling_column_name,
)
from fantasy_football.modelling.defender import (
    POSITION as DEFENDER_POSITION,
)
from fantasy_football.modelling.defender import (
    PRODUCTION_ALIAS as DEFENDER_ALIAS,
)
from fantasy_football.modelling.defender import (
    REGISTERED_MODEL as DEFENDER_REGISTERED_MODEL,
)
from fantasy_football.modelling.models import (
    MODELS_BY_POSITION,
    PointsModel,
    StoredPredictionModel,
)
from fantasy_football.storage.tables import FORWARD_KIND, POINTS_PREDICTION

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
        One row per ``player_code``, with a ``baseline`` column and a
        ``baseline_season`` column naming the season that baseline was taken
        from (so callers can see how stale it is).

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
            schema={
                "player_code": pl.Int64,
                "baseline": pl.Float64,
                "baseline_season": pl.Utf8,
            }
        )
    return (
        eligible.sort(["season", "gw"])
        .group_by("player_code")
        .agg(
            pl.col(rolling_col).last().alias("baseline"),
            pl.col("season").last().alias("baseline_season"),
        )
    )


def _log_stale_baselines(
    joined: pl.DataFrame, current_season: str, worst_n: int = 10
) -> None:
    """Log how many baselines predate the immediately-preceding season.

    A cross-season baseline lookup will happily reach years back for a player
    who has not featured since -- six-year-old form then gets used verbatim as
    current form. Nothing here changes which baseline is chosen; this only
    makes the staleness visible.

    Parameters
    ----------
    joined : pl.DataFrame
        Roster rows joined to their baselines, carrying ``name`` and
        ``baseline_season``.
    current_season : str
        The season being predicted.
    worst_n : int, optional
        How many of the oldest offenders to name in the log line. Defaults to
        10.

    Returns
    -------
    None
    """
    previous = previous_season(current_season)
    stale = joined.filter(
        pl.col("baseline_season").is_not_null()
        & (pl.col("baseline_season") < previous)
    ).sort(["baseline_season", "name"])
    if stale.is_empty():
        return
    worst = [
        f"{name} ({season})"
        for name, season in zip(
            stale["name"].to_list()[:worst_n],
            stale["baseline_season"].to_list()[:worst_n],
            strict=True,
        )
    ]
    logger.info(
        "%d of %d players draw their baseline from a season older than %s; "
        "oldest first: %s",
        stale.height,
        joined.height,
        previous,
        worst,
    )


def _dedupe_by_name(baselines: pl.DataFrame) -> pl.DataFrame:
    """Collapse same-named players to one row, deterministically and loudly.

    Downstream keys on ``name``, so two players sharing
    ``first_name || ' ' || second_name`` would otherwise become one variable
    with last-write-wins. The row with the lowest ``element`` survives, which
    makes the choice reproducible run to run rather than dependent on join
    order.

    Parameters
    ----------
    baselines : pl.DataFrame
        Baseline rows carrying ``name`` and ``element``.

    Returns
    -------
    pl.DataFrame
        The same frame with at most one row per ``name``.
    """
    duplicated = (
        baselines.group_by("name")
        .agg(pl.len().alias("rows"))
        .filter(pl.col("rows") > 1)
    )
    if not duplicated.is_empty():
        logger.warning(
            "%d display name(s) are shared by more than one player; keeping "
            "the lowest element id for each and dropping the rest: %s",
            duplicated.height,
            sorted(duplicated["name"].to_list()),
        )
    return baselines.sort("element").unique(
        subset=["name"], keep="first", maintain_order=True
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
        ``team``, ``element`` and ``baseline``. One row per ``name``:
        everything downstream (``optimiser._build_problem``,
        ``optimiser._load_prices``) keys on ``name``, so two players sharing a
        display name would otherwise collapse into a single MILP variable with
        last-write-wins. Both branches therefore deduplicate on ``name``; on
        the roster branch the survivor is deterministic -- the row with the
        lowest ``element`` wins -- and the losers are logged.
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
        _log_stale_baselines(joined, current_season)
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
        teamless = sorted(
            filled.filter(pl.col("team").is_null())["name"].to_list()
        )
        for name in teamless:
            logger.debug("Dropping player %r — no team in the roster", name)
        filled = filled.filter(pl.col("team").is_not_null())
        return _dedupe_by_name(
            filled.select("name", "position", "team", "element", "baseline")
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


def _apply_models(
    fx: pl.DataFrame,
    baselines: pl.DataFrame,
    models: dict[str, PointsModel] | None = None,
) -> pl.DataFrame:
    """Join baselines, build factors, and apply the per-position model.

    Parameters
    ----------
    fx : pl.DataFrame
        Fixtures enriched with player and opponent ELO.
    baselines : pl.DataFrame
        One row per player with their latest rolling baseline and team.
    models : dict[str, PointsModel] | None, optional
        Position-to-model map. Defaults to ``MODELS_BY_POSITION``.
        Passed explicitly by :func:`_predict` so a position served by a
        registered model can be swapped in per run.

    Returns
    -------
    pl.DataFrame
        Per-(player, future_gw) rows with a ``predicted_points`` column.
    """
    models = MODELS_BY_POSITION if models is None else models
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
    for position, model in models.items():
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


class MissingProductionModelError(RuntimeError):
    """Raised when a position has no live model to serve predictions."""


class UncoveredPositionError(RuntimeError):
    """Raised when a model-served position leaves rows unpredicted."""


# How many uncovered rows the error names before it stops listing them.
UNCOVERED_SAMPLE_N = 10


def _check_defender_coverage(scored: pl.DataFrame) -> None:
    """Raise if any defender row came back without a prediction.

    :class:`~fantasy_football.modelling.models.StoredPredictionModel`
    returns null where nothing is stored, and there is deliberately no
    per-row fallback to the formula -- so a gap has to fail rather than
    degrade. Without this it fails anyway, but as a ``TypeError`` raised
    inside pulp while building the LP, which names neither the position
    nor the rows.

    Parameters
    ----------
    scored : pl.DataFrame
        Output of :func:`_apply_models`, carrying ``position``,
        ``predicted_points``, ``season``, ``gw`` and ``player_id``.

    Raises
    ------
    UncoveredPositionError
        When a ``DEF`` row has a null ``predicted_points``.
    """
    uncovered = scored.filter(
        (pl.col("position") == DEFENDER_POSITION)
        & pl.col("predicted_points").is_null()
    ).sort(["season", "gw", "player_id"])
    if uncovered.is_empty():
        return
    sample = uncovered.select("season", "gw", "player_id").rows()[
        :UNCOVERED_SAMPLE_N
    ]
    raise UncoveredPositionError(
        f"{uncovered.height} {DEFENDER_POSITION} row(s) have no stored "
        f"prediction, e.g. {sample} as (season, gw, element). "
        f"{DEFENDER_POSITION} is served by {DEFENDER_REGISTERED_MODEL} "
        f"with no per-row fallback, so this is a coverage gap in the "
        f"forward scoring run, not a degradation to absorb. Re-run "
        f"score_forward_defender_points for these gameweeks."
    )


def load_forward_defender_predictions() -> pl.DataFrame | None:
    """Return stored forward defender predictions, or None if there are none.

    Returns
    -------
    pl.DataFrame | None
        Match-grain rows with ``season``, ``gw``, ``element`` and
        ``predicted_points``, or ``None`` when nothing is stored.
    """
    stored = POINTS_PREDICTION.load().filter(
        (pl.col("prediction_kind") == FORWARD_KIND)
        & (pl.col("position") == DEFENDER_POSITION)
    )
    if stored.is_empty():
        return None
    return stored.select("season", "gw", "element", "predicted_points")


def _models_for_run(is_backtest: bool) -> dict[str, PointsModel]:
    """Return the position-to-model map this run should score with.

    Live runs serve ``DEF`` from the registered model's stored forward
    predictions, and fail loudly when none exist -- see
    :class:`MissingProductionModelError`.

    Backtests deliberately keep ``DEF`` on
    :class:`~fantasy_football.modelling.models.RollingFormulaModel`. The
    rolling-origin harness replays *past* gameweeks, but the stored
    predictions are forward-kind and cover only *unplayed* ones, so a
    replayed pivot would match none of them. That would not surface as
    an obvious gap either: ``evaluation`` sums ``predicted_points`` per
    player-gameweek, and a polars ``sum`` over an all-null group returns
    ``0.0``, so every historical defender would reach the metrics as a
    numeric zero -- MAE and RMSE measuring distance from zero, a
    strongly negative skill score, all of it logged to the DEF
    experiment as though it described the model. Wiring the harness to
    the defender model means teaching it to read the backfill-kind rows
    and reworking the replay; that is deferred, and until it lands the
    honest behaviour is the one the spec already documents -- the
    backtest measures the formula.

    Parameters
    ----------
    is_backtest : bool
        Whether this is a historical replay rather than a live run.

    Returns
    -------
    dict[str, PointsModel]
        ``MODELS_BY_POSITION`` for a backtest; the same map with ``DEF``
        swapped for a :class:`StoredPredictionModel` for a live run.

    Raises
    ------
    MissingProductionModelError
        On a live run with no stored forward defender predictions.
    """
    if is_backtest:
        logger.info(
            "Backtest run: scoring %s with the rolling formula, not the "
            "registered model. Stored predictions are forward-kind and "
            "cover only unplayed gameweeks, so they cannot serve a "
            "replayed pivot. Do not read %s metrics from this run as "
            "measuring the ML model.",
            DEFENDER_POSITION,
            DEFENDER_POSITION,
        )
        return MODELS_BY_POSITION

    defender_predictions = load_forward_defender_predictions()
    if defender_predictions is None:
        raise MissingProductionModelError(
            f"No forward {DEFENDER_POSITION} predictions in "
            f"points_prediction. Train a model, then promote a version to "
            f"'{DEFENDER_ALIAS}' on {DEFENDER_REGISTERED_MODEL} in the "
            f"MLflow UI. The optimiser needs five defenders, so continuing "
            f"would hand it an infeasible squad problem."
        )
    return MODELS_BY_POSITION | {
        DEFENDER_POSITION: StoredPredictionModel(defender_predictions)
    }


def _predict(
    current_season: str,
    horizon_n: int | None = None,
    as_of_gw: int | None = None,
    is_backtest: bool = False,
) -> pl.DataFrame:
    """Produce per-(player, future_gw) point predictions and return them.

    Loads the three input tables: rolling points, fixtures, and team ELO, then
    gets each player's baseline (latest current rolling points) and figures
    out which gameweeks require prediction. If `horizon_n` is not provided, it
    will predict on all future gameweeks. It then attaches ELO ratings for
    each fixture and calculates the predicted points for each player.

    Whether the FPL player snapshot may be read is decided by ``is_backtest``
    alone, never by ``as_of_gw``. The two carry different meanings and a live
    run legitimately sets both:

    * ``main.main`` is a live run planning from a gameweek. It leaves
      ``is_backtest`` False, so the snapshot -- a capture of *now* -- is the
      right source of truth for who plays for whom, whether or not it also
      passes ``as_of_gw`` (which it does, derived from the team file, and
      which is 0 for a pre-season GW1 team). Before a ball is kicked the
      roster is the *only* source of players, so suppressing it here is what
      produced an empty predictions.csv.
    * ``evaluation._collect_predictions_vs_actuals`` replays past pivots and
      passes ``is_backtest=True``. Reading a present-day snapshot there would
      leak future club membership into a past prediction, so the legacy
      player-week-derived identity path is used instead.

    Parameters
    ----------
    current_season: str
        The current season.
    horizon_n: int | None
        The number of future gameweeks to predict. If None, all future gameweeks
        will be predicted.
    as_of_gw : int | None
        Pivot gameweek. When set, predictions cover gameweeks after as_of_gw
        and baselines use only form at or before it. When None, the pivot is
        the latest completed current-season gameweek. Defaults to None.
    is_backtest : bool, optional
        Whether this is a historical replay rather than a live run. True
        suppresses every present-day source -- currently the player snapshot
        that backs the roster -- so a past pivot cannot see the present.
        It also keeps ``DEF`` on the rolling formula rather than the
        registered model, for the reasons in :func:`_models_for_run`.
        Defaults to False, which is what live callers want.

    Returns
    -------
    pl.DataFrame
        The predictions DataFrame.

    Raises
    ------
    MissingProductionModelError
        When this is a live run and no forward defender predictions are
        stored. Defenders are served by a registered model, and there is
        deliberately no per-row fallback to the formula: mixing
        formula-scored and model-scored defenders in one column would
        put two uncalibrated scales side by side and make the
        optimiser's comparison between them meaningless. So the choice
        is made once, here, for the whole run.
    UncoveredPositionError
        When this is a live run and a defender reaches the end of
        scoring with no stored prediction. See
        :func:`_check_defender_coverage`.
    """
    models = _models_for_run(is_backtest)
    rolling = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("rolling_points.csv"),
        try_parse_dates=True,
    )
    roster = None if is_backtest else current_roster(current_season)
    baselines = _baselines(rolling, current_season, as_of_gw, roster)
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

    # team_elo carries one row per FPL alias, so a club known by two names
    # (e.g. Ipswich) appears twice with an identical rating over an identical
    # interval. Deduplicate on the rating interval before taking the median,
    # or the multi-alias clubs get double weight in the fallback value.
    median_elo = team_elo.unique(subset=["elo", "from_date", "to_date"])[
        "elo"
    ].median()
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

    fx = _apply_models(fx, baselines, models)
    # Live runs only: on the backtest path DEF is formula-scored, so a
    # null there means something else entirely and is not this check's
    # business. See _models_for_run.
    if not is_backtest:
        _check_defender_coverage(fx)
    return fx


def predict_points(
    current_season: str,
    horizon_n: int | None = None,
    as_of_gw: int | None = None,
    is_backtest: bool = False,
) -> pl.DataFrame:
    """Compute predictions, write them to predictions.csv, and return them.

    Thin IO wrapper around :func:`_predict`. See that function for the
    prediction logic and parameter meanings.

    This is the live entry point: ``main.main`` calls it and leaves
    ``is_backtest`` at its default of False, so the player snapshot supplies
    the roster even when ``as_of_gw`` is set from a team file. The backtest
    harness in :mod:`fantasy_football.modelling.evaluation` bypasses this
    wrapper and calls :func:`_predict` with ``is_backtest=True``.

    Parameters
    ----------
    current_season : str
        The current season.
    horizon_n : int | None
        Number of future gameweeks to predict. None predicts all of them.
    as_of_gw : int | None
        Pivot gameweek: predictions cover gameweeks after it, and baselines
        use only form at or before it.
    is_backtest : bool, optional
        Whether this is a historical replay. True suppresses present-day
        sources (the snapshot-derived roster) so a past pivot cannot see the
        present. Defaults to False for live callers.

    Returns
    -------
    pl.DataFrame
        The predictions DataFrame.
    """
    predictions = _predict(current_season, horizon_n, as_of_gw, is_backtest)
    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    predictions.write_csv(TRANSFORMED_DATA_FOLDER.joinpath("predictions.csv"))
    logger.info(
        "Wrote %d predictions covering gws %s",
        predictions.height,
        sorted(set(predictions["gw"].to_list())),
    )
    return predictions
