"""Pure regression metrics for scoring points predictions.

None of these functions touch MLflow, files, or position grouping. Callers
slice their data by position and pass plain sequences or a small DataFrame.
"""

from collections.abc import Sequence

import numpy as np
import polars as pl
from sklearn.metrics import mean_poisson_deviance

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


def spearman_by_gw(
    df: pl.DataFrame,
    gw_col: str = "gw",
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
    gw_col : str
        Name of the gameweek column.
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
    for (_gw,), sub in df.group_by([gw_col], maintain_order=True):
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
    gw_col: str = "gw",
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
    gw_col : str
        Name of the gameweek column.
    id_col : str
        Name of the player ID column.

    Returns
    -------
    float
        Mean per-gameweek precision@k.
    """
    precisions: list[float] = []
    for (_gw,), sub in df.group_by([gw_col], maintain_order=True):
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
