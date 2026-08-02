"""Expanding-window cross-validation folds.

Both helpers train on everything before a point in time and test on the
next unseen slice, which is what deployment does. They differ only in the
unit of time.

``season_folds`` is right when a model has several complete seasons and
the season boundary is the meaningful break -- the minutes classifier.
``gameweek_folds`` is right when it does not: the defender model's full
feature set exists for one complete season plus the current partial one,
so season folds would yield a single holdout rather than a distribution.
"""

from collections.abc import Iterable


def season_folds(seasons: list[str]) -> list[tuple[list[str], str]]:
    """Build expanding-window CV folds over sorted seasons.

    Each fold trains on every prior season and tests on the next unseen
    one, mirroring deployment. Requires at least two seasons.

    Parameters
    ----------
    seasons : list[str]
        Season strings; sorted ascending internally.

    Returns
    -------
    list[tuple[list[str], str]]
        ``(train_seasons, test_season)`` pairs.
    """
    ordered = sorted(seasons)
    return [(ordered[:i], ordered[i]) for i in range(1, len(ordered))]


def gameweek_folds(
    keys: Iterable[tuple[str, int]], min_train_gws: int = 10
) -> list[tuple[list[tuple[str, int]], tuple[str, int]]]:
    """Build expanding-window CV folds over ``(season, gw)`` keys.

    Keys are deduplicated and sorted, which orders them chronologically
    because season strings sort lexicographically in calendar order
    ("2025-26" < "2026-27") and gameweeks sort numerically within one.
    Each fold trains on every prior gameweek and tests on the next.

    ``min_train_gws`` exists because the earliest folds are noise: a
    model trained on three gameweeks of rolling-form features, most of
    which have empty windows, says nothing useful about the model.

    Parameters
    ----------
    keys : Iterable[tuple[str, int]]
        ``(season, gw)`` pairs present in the model frame.
    min_train_gws : int, optional
        Minimum gameweeks in the training side of the first fold.

    Returns
    -------
    list[tuple[list[tuple[str, int]], tuple[str, int]]]
        ``(train_keys, test_key)`` pairs. Empty when there are not
        enough gameweeks to form one fold.
    """
    ordered = sorted(set(keys))
    return [
        (ordered[:i], ordered[i]) for i in range(min_train_gws, len(ordered))
    ]
