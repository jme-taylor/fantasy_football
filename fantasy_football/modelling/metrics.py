"""Pure regression metrics for scoring points predictions.

None of these functions touch MLflow, files, or position grouping. Callers
slice their data by position and pass plain sequences or a small DataFrame.
"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import polars as pl
from sklearn.metrics import mean_poisson_deviance

logger = logging.getLogger(__name__)

_POISSON_FLOOR = 1e-6


def mae(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Mean absolute error.

    Parameters
    ----------
    predicted : Sequence[float]
        Predicted points.
    actual : Sequence[float]
        Actual points.

    Returns
    -------
    float
        Mean absolute error.
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    return float(np.mean(np.abs(p - a)))


def rmse(predicted: Sequence[float], actual: Sequence[float]) -> float:
    """Root mean squared error.

    Parameters
    ----------
    predicted : Sequence[float]
        Predicted points.
    actual : Sequence[float]
        Actual points.

    Returns
    -------
    float
        Root mean squared error.
    """
    p = np.asarray(predicted, dtype=float)
    a = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.mean((p - a) ** 2)))


def skill_score(
    predicted: Sequence[float],
    actual: Sequence[float],
    baseline: Sequence[float],
) -> float:
    """1 - MAE(model) / MAE(baseline).

    Positive means the model beats the naive baseline. Returns nan when the
    baseline error is zero (nothing to improve on).

    Parameters
    ----------
    predicted : Sequence[float]
        Predicted points.
    actual : Sequence[float]
        Actual points.
    baseline : Sequence[float]
        Baseline points.

    Returns
    -------
    float
        Skill score.
    """
    baseline_mae = mae(baseline, actual)
    if baseline_mae == 0:
        return float("nan")
    return 1.0 - mae(predicted, actual) / baseline_mae


def poisson_deviance(
    predicted: Sequence[float], actual: Sequence[float]
) -> float:
    """Mean Poisson deviance; predictions are floored to stay positive.

    Actuals are floored at 0.0 because Poisson deviance is defined only for
    non-negative targets; rare negative FPL gameweek totals are clipped to 0.
    """
    p = np.clip(np.asarray(predicted, dtype=float), _POISSON_FLOOR, None)
    a = np.clip(np.asarray(actual, dtype=float), 0.0, None)
    return float(mean_poisson_deviance(a, p))


def count_poisson_deviance(
    actual: "Sequence[float] | np.ndarray",
    expected: "Sequence[float] | np.ndarray",
) -> float:
    """Mean Poisson deviance of expected counts against realised ones.

    Written out rather than taken from sklearn so the zero-count rows --
    which are almost all of them on a rare-event head -- contribute their
    ``2 * expected`` term rather than a nan from ``0 * log(0)``.

    Parameters
    ----------
    actual : Sequence[float]
        Realised counts.
    expected : Sequence[float]
        Expected counts, floored to stay positive.

    Returns
    -------
    float
        Mean Poisson deviance.
    """
    counts = np.asarray(actual, dtype=float)
    safe = np.clip(np.asarray(expected, dtype=float), 1e-9, None)
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(counts > 0, counts * np.log(counts / safe), 0.0)
    return float(np.mean(2.0 * (term - (counts - safe))))


def top_decile_ratio(
    rate: "Sequence[float] | np.ndarray",
    actual: "Sequence[float] | np.ndarray",
    expected: "Sequence[float] | np.ndarray",
) -> float:
    """Return predicted over actual counts in the top predicted decile.

    Below one means the head under-predicts its own best players, which
    is the compression a squared-error forest is expected to show -- and
    that decile is the one holding every player worth captaining.

    Parameters
    ----------
    rate : Sequence[float]
        Predicted per-90 rates, which decide the decile.
    actual : Sequence[float]
        Realised counts.
    expected : Sequence[float]
        Expected counts.

    Returns
    -------
    float
        The ratio, or nan when there is nothing to rank or the decile
        realised no events.
    """
    rates = np.asarray(rate, dtype=float)
    if rates.size == 0:
        return float("nan")
    top = rates >= float(np.quantile(rates, 0.9))
    realised = float(np.sum(np.asarray(actual, dtype=float)[top]))
    if realised == 0.0:
        return float("nan")
    return float(np.sum(np.asarray(expected, dtype=float)[top]) / realised)


def spearman_by_gw(
    df: pl.DataFrame,
    gw_cols: Sequence[str] = ("gw",),
    pred_col: str = "predicted_points",
    actual_col: str = "actual",
) -> float:
    """Mean per-gameweek Spearman rank correlation.

    Computes Spearman (Pearson on average-ranks) within each gameweek, then
    averages across gameweeks. Gameweeks with fewer than two players, or with
    no variation in predictions or actuals, are skipped. Returns nan when no
    gameweek qualifies.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the predicted and actual points.
    gw_cols : Sequence[str]
        Columns identifying one gameweek. Pass ``("season", "gw")`` for
        any frame spanning more than one season: gameweek numbers repeat
        each year, so grouping on ``gw`` alone would rank two seasons'
        players against each other.
    pred_col : str
        Name of the predicted points column.
    actual_col : str
        Name of the actual points column.

    Returns
    -------
    float
        Mean per-gameweek Spearman rank correlation.
    """
    correlations: list[float] = []
    for _key, sub in df.group_by(list(gw_cols), maintain_order=True):
        if sub.height < 2:
            continue
        pred_rank = sub[pred_col].rank(method="average")
        actual_rank = sub[actual_col].rank(method="average")
        if pred_rank.n_unique() == 1 or actual_rank.n_unique() == 1:
            continue
        corr = (
            pl.DataFrame({"p": pred_rank, "a": actual_rank})
            .select(pl.corr("p", "a"))
            .item()
        )
        if corr is not None:
            correlations.append(corr)
    if not correlations:
        return float("nan")
    return float(np.mean(correlations))


def precision_at_k(
    df: pl.DataFrame,
    k: int,
    gw_cols: Sequence[str] = ("gw",),
    id_col: str = "player_id",
    pred_col: str = "predicted_points",
    actual_col: str = "actual",
) -> float:
    """Mean per-gameweek precision@k.

    For each gameweek: of the model's top-k predicted players, the fraction
    that are also in the actual top-k. ``k`` is clamped to the number of
    players available in the gameweek. Returns nan when no gameweek qualifies.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame containing the predicted and actual points.
    k : int
        The number of players to consider.
    gw_cols : Sequence[str]
        Columns identifying one gameweek. Pass ``("season", "gw")`` for
        any frame spanning more than one season.
    id_col : str
        Name of the player ID column.
    pred_col : str
        Name of the predicted points column.
    actual_col : str
        Name of the actual points column.

    Returns
    -------
    float
        Mean per-gameweek precision@k.
    """
    precisions: list[float] = []
    for _key, sub in df.group_by(list(gw_cols), maintain_order=True):
        effective_k = min(k, sub.height)
        if effective_k == 0:
            continue
        top_pred = set(
            sub.sort(pred_col, descending=True)
            .head(effective_k)[id_col]
            .to_list()
        )
        top_actual = set(
            sub.sort(actual_col, descending=True)
            .head(effective_k)[id_col]
            .to_list()
        )
        precisions.append(len(top_pred & top_actual) / effective_k)
    if not precisions:
        return float("nan")
    return float(np.mean(precisions))


class Metrics(Protocol):
    """One fold's scores, flattenable for MLflow.

    Every predictor's fold scoring returns something satisfying this, so
    :func:`aggregate` and the MLflow logging in
    :class:`fantasy_football.modelling.predictor.Predictor` stay ignorant
    of which model produced the numbers.
    """

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, one level deep."""
        ...


@dataclass(frozen=True, slots=True)
class PointsMetrics:
    """Scores for a points regressor over one fold.

    Shared by every position. ``k`` is deliberately absent: it varies by
    position, so carrying it here would give each position a different
    metric name and make MLflow runs incomparable. Log it as a param.
    """

    mae: float
    rmse: float
    skill_score: float
    spearman: float
    precision_at_k: float

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, one level deep."""
        return asdict(self)


@dataclass(frozen=True, slots=True)
class MinutesMetrics:
    """Scores for the minutes classifier over one fold.

    The two AUCs are optional because a fold whose test split has only one
    class present cannot have an AUC at all. They are dropped from
    :meth:`as_dict` when absent rather than logged as nan, which keeps
    :func:`aggregate` from producing a degenerate mean.
    """

    logloss_appear: float
    brier_appear: float
    logloss_60: float
    brier_60: float
    e_min_mae: float
    auc_appear: float | None = None
    auc_60: float | None = None

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, omitting absent AUCs."""
        return {
            key: value
            for key, value in asdict(self).items()
            if value is not None
        }


def aggregate(per_fold: Sequence[Metrics]) -> dict[str, float]:
    """Mean and standard deviation of every metric across folds.

    Only metrics present in *every* fold are aggregated, so a metric that
    one fold could not compute does not silently average over a subset.

    Parameters
    ----------
    per_fold : Sequence[Metrics]
        One entry per scored fold.

    Returns
    -------
    dict[str, float]
        ``{metric}_mean`` and ``{metric}_std`` for each shared metric.
        Empty when ``per_fold`` is empty.
    """
    if not per_fold:
        return {}
    rows = [metrics.as_dict() for metrics in per_fold]
    shared = set(rows[0]).intersection(*(set(row) for row in rows[1:]))
    agg: dict[str, float] = {}
    for key in sorted(shared):
        values = [row[key] for row in rows]
        n_nan = sum(1 for value in values if np.isnan(value))
        if n_nan:
            logger.warning(
                "%d of %d folds had a nan %s; the aggregate is degenerate.",
                n_nan,
                len(values),
                key,
            )
        agg[f"{key}_mean"] = float(np.mean(values))
        agg[f"{key}_std"] = float(np.std(values))
    return agg


def metric_name(prefix: str | None, key: str) -> str:
    """Name one metric for MLflow, under its strategy's prefix.

    Parameters
    ----------
    prefix : str | None
        The fold strategy's ``metric_prefix``. None leaves the key bare.
    key : str
        The metric's own name.

    Returns
    -------
    str
        The name to log under.
    """
    return key if prefix is None else f"{prefix}_{key}"


def summarise(
    per_fold: Sequence[Metrics], prefix: str | None, aggregated: bool
) -> dict[str, float]:
    """Name a run's scores for MLflow, under its strategy's prefix.

    The prefix keeps a mean of per-gameweek errors out of the same MLflow
    column as a single pooled error: they are different quantities and
    would otherwise be compared by eye. A strategy scoring one fold needs
    no prefix, since the aggregating strategies carry ``_mean``/``_std``
    that already tell the two apart.

    ``aggregated`` is the strategy's own answer, not a count of the folds
    that happened to survive. A cross-validating strategy always reports
    ``_mean``/``_std``, even in a season short enough to leave it one
    testable fold, so a run never silently drops out of a query keyed on
    the aggregate names. A holdout is one measurement and reports its
    metrics directly, since a standard deviation of zero would read as a
    perfectly stable model rather than as an absent spread.

    Parameters
    ----------
    per_fold : Sequence[Metrics]
        One entry per scored fold.
    prefix : str | None
        The fold strategy's ``metric_prefix``.
    aggregated : bool
        The fold strategy's ``aggregates``.

    Returns
    -------
    dict[str, float]
        Prefixed metric names and values. Empty when nothing was scored.
    """
    if not per_fold:
        return {}
    scores = aggregate(per_fold) if aggregated else per_fold[0].as_dict()
    return {metric_name(prefix, key): value for key, value in scores.items()}
