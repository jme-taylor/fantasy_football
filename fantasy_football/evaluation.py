"""Rolling-origin evaluation of the per-position points models.

For every completed gameweek across all available seasons, predict the next
gameweek one step ahead, join to the actual points, and score each position
with the metrics in :mod:`fantasy_football.metrics`. Results are logged to
MLflow, one experiment per position.
"""

import logging

import mlflow
import polars as pl

from fantasy_football import metrics
from fantasy_football.constants import (
    EXPERIMENT_BY_POSITION,
    MLFLOW_TRACKING_URI,
    PRECISION_K_BY_POSITION,
    ROLLING_WINDOW,
    TRANSFORMED_DATA_FOLDER,
)
from fantasy_football.data_transformation import KNOWN_POSITIONS
from fantasy_football.prediction import _predict

logger = logging.getLogger(__name__)


def _select_pivots(
    rolling: pl.DataFrame, season: str, rolling_window: int
) -> list[int]:
    """Return pivot gameweeks for one season.

    A pivot needs at least ``rolling_window`` earlier gameweeks (so the
    baseline is defined) and a following gameweek (so an actual exists to
    score against).

    Parameters
    ----------
    rolling : pl.DataFrame
        The rolling-points table.
    season : str
        Season to select pivots for.
    rolling_window : int
        Required number of prior gameweeks.

    Returns
    -------
    list[int]
        Sorted pivot gameweeks.
    """
    gws = sorted(
        rolling.filter(pl.col("season") == season)["gw"].unique().to_list()
    )
    if not gws:
        return []
    last_gw = gws[-1]
    pivots = []
    for index, gw in enumerate(gws):
        prior = index  # number of earlier gameweeks
        if prior >= rolling_window and gw < last_gw:
            pivots.append(gw)
    return pivots


def _actuals(rolling: pl.DataFrame) -> pl.DataFrame:
    """One actual total_points per (season, player_id, gw)."""
    return (
        rolling.select(
            "season",
            pl.col("element").alias("player_id"),
            "gw",
            "total_points",
        )
        .group_by("season", "player_id", "gw")
        .agg(pl.col("total_points").sum().alias("actual"))
    )


def _collect_predictions_vs_actuals(rolling_window: int) -> pl.DataFrame:
    """Replay every pivot across all seasons and join predictions to actuals.

    Parameters
    ----------
    rolling_window : int
        Required prior-gameweek history for a pivot.

    Returns
    -------
    pl.DataFrame
        One row per (season, player_id, gw) with ``position``,
        ``predicted_points``, ``baseline`` and ``actual``.
    """
    rolling = pl.read_csv(
        TRANSFORMED_DATA_FOLDER.joinpath("rolling_points.csv"),
        try_parse_dates=True,
    )
    actuals = _actuals(rolling)
    seasons = sorted(rolling["season"].unique().to_list())

    prediction_frames = []
    for season in seasons:
        for pivot in _select_pivots(rolling, season, rolling_window):
            preds = _predict(season, horizon_n=1, as_of_gw=pivot)
            if preds.is_empty():
                continue
            prediction_frames.append(preds)

    if not prediction_frames:
        return pl.DataFrame()

    predictions = pl.concat(prediction_frames, how="vertical")
    # Sum across fixtures so a double gameweek is one row per player-gw.
    predictions = predictions.group_by(
        "season", "player_id", "gw", "position"
    ).agg(
        pl.col("predicted_points").sum().alias("predicted_points"),
        pl.col("baseline").first().alias("baseline"),
    )
    return predictions.join(
        actuals, on=["season", "player_id", "gw"], how="inner"
    )


def _metrics_for_position(
    df: pl.DataFrame, position: str
) -> dict[str, float | int]:
    """Compute the six metrics for one position's collected rows."""
    df = df.with_columns(
        (pl.col("season") + "-" + pl.col("gw").cast(pl.Utf8)).alias(
            "season_gw"
        )
    )
    predicted = df["predicted_points"].to_list()
    actual = df["actual"].to_list()
    baseline = df["baseline"].to_list()
    k = PRECISION_K_BY_POSITION[position]
    return {
        "skill_score": metrics.skill_score(predicted, actual, baseline),
        "spearman": metrics.spearman_by_gw(df, gw_col="season_gw"),
        "precision_at_k": metrics.precision_at_k(df, k=k, gw_col="season_gw"),
        "mae": metrics.mae(predicted, actual),
        "rmse": metrics.rmse(predicted, actual),
        "poisson_deviance": metrics.poisson_deviance(predicted, actual),
        "n_samples": df.height,
    }


def evaluate(
    rolling_window: int = ROLLING_WINDOW,
) -> dict[str, dict[str, float | int]]:
    """Score every position over the full historical record.

    Parameters
    ----------
    rolling_window : int
        Required prior-gameweek history for a pivot (defaults to the project's
        ROLLING_WINDOW; passed explicitly to keep this function pure-ish).

    Returns
    -------
    dict[str, dict]
        Metrics keyed by position, only for positions present in the data.
    """
    collected = _collect_predictions_vs_actuals(rolling_window)
    if collected.is_empty():
        logger.warning("No predictions collected; nothing to evaluate")
        return {}

    results = {}
    for position in KNOWN_POSITIONS:
        sub = collected.filter(pl.col("position") == position)
        if sub.is_empty():
            continue
        results[position] = _metrics_for_position(sub, position)
        logger.info(
            "Scored %s over %d samples: %s",
            position,
            sub.height,
            results[position],
        )
    return results


def log_results_to_mlflow(
    results: dict[str, dict[str, float | int]],
    rolling_window: int = ROLLING_WINDOW,
) -> None:
    """Log each position's metrics to its own MLflow experiment.

    Parameters
    ----------
    results : dict[str, dict]
        Output of :func:`evaluate`.
    rolling_window : int
        Logged as a run parameter.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    for position, position_metrics in results.items():
        mlflow.set_experiment(EXPERIMENT_BY_POSITION[position])
        with mlflow.start_run():
            mlflow.log_params(
                {
                    "horizon": 1,
                    "rolling_window": rolling_window,
                    "n_samples": position_metrics["n_samples"],
                }
            )
            mlflow.log_metrics(
                {k: v for k, v in position_metrics.items() if k != "n_samples"}
            )


def run_evaluation(
    rolling_window: int = ROLLING_WINDOW,
) -> dict[str, dict[str, float | int]]:
    """Evaluate every position and log the results to MLflow.

    Parameters
    ----------
    rolling_window : int
        Required prior-gameweek history for a pivot (defaults to the project's
        ROLLING_WINDOW; passed explicitly to keep this function pure-ish).

    Returns
    -------
    dict[str, dict]
        Metrics keyed by position, only for positions present in the data.
    """
    results = evaluate(rolling_window)
    log_results_to_mlflow(results, rolling_window)
    return results
