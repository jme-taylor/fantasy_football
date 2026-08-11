import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import polars as pl

logger = logging.getLogger(__name__)

DEFAULT_MIN_TRAIN_GAMEWEEKS = 10


@dataclass
class FoldTestKey:
    """Key for filtering the test data in a fold."""

    season: str
    gw: int | None = None  # None means the entire season


@dataclass
class Fold:
    """A fold of training and test data."""

    train: pl.DataFrame
    test: pl.DataFrame
    test_key: FoldTestKey


class FoldStrategy(Protocol):
    """Protocol for a fold strategy.

    Any fold strategy must take a polars dataframe of training data spilt
    it into a a sequence of folds.
    """

    def split(self, training_data: pl.DataFrame) -> Iterable[Fold]:
        """Split the training data into folds.

        Parameters
        ----------
        training_data : pl.DataFrame
            The training data to split into folds.

        Returns
        -------
        Iterable[Fold]
            The folds.
        """
        ...


class SeasonFoldStrategy:
    """Fold strategy that splits the training data into folds by season."""

    def split(self, training_data: pl.DataFrame) -> Iterable[Fold]:
        """Split the training data into folds by season.

        Each fold is a training set of all seasons up to the test season
        for that fold and a test set of the test season for that fold.

        Parameters
        ----------
        training_data : pl.DataFrame
            The training data to split into folds.

        Returns
        -------
        Iterable[Fold]
            The folds.
        """
        unique_seasons = sorted(training_data["season"].unique().to_list())
        return [
            Fold(
                train=training_data.filter(
                    pl.col("season").is_in(unique_seasons[:i])
                ),
                test=training_data.filter(
                    pl.col("season") == unique_seasons[i]
                ),
                test_key=FoldTestKey(season=unique_seasons[i]),
            )
            for i in range(1, len(unique_seasons))
        ]


class ExpandingGameweekFoldStrategy:
    """Expanding-window folds at ``(season, gw)`` grain.

    Right when a model does not have several complete seasons to hold
    out: the defender model's full feature set exists for one complete
    season plus the current partial one, so season folds would yield a
    single holdout rather than a distribution.

    Parameters
    ----------
    min_train_folds : int | None, optional
        Gameweeks the first fold must train on. Below roughly ten, most
        rolling-form windows are still empty and the fold measures noise.
    test_seasons : Sequence[str] | None, optional
        Seasons a fold is allowed to *test* on. Defaults to every season
        in the frame. Pass the seasons in which every model feature is
        actually published: a fold testing an earlier gameweek scores a
        model the imputer has silently stripped down, because features
        that are entirely null in the training slice get dropped and the
        model fits on what remains. Such folds measure a different model
        than the one being registered, and averaging them into the
        aggregate buries the signal the promotion decision needs.

        Only the test side is restricted. A restricted fold still trains
        on every earlier gameweek, uncovered seasons included, which is
        what the deployed model does.
    """

    def __init__(
        self,
        min_train_folds: int | None = None,
        test_seasons: Sequence[str] | None = None,
    ) -> None:
        """Initialize the expanding gameweek fold strategy.

        Parameters
        ----------
        min_train_folds : int | None, optional
            Gameweeks the first fold must train on. Below roughly ten, most
            rolling-form windows are still empty and the fold measures noise.
            Will error if less than 1.
        test_seasons : Sequence[str] | None, optional
            Seasons a fold is allowed to *test* on. Defaults to every season
            in the frame. Pass the seasons in which every model feature is
            actually published.
        """
        if min_train_folds is not None and min_train_folds < 1:
            raise ValueError("min_train_folds must be at least 1")
        if test_seasons is not None and not test_seasons:
            raise ValueError(
                "test_seasons must name at least one season; pass None to "
                "test on every season in the frame"
            )
        self.min_train_folds = min_train_folds or DEFAULT_MIN_TRAIN_GAMEWEEKS
        self.test_seasons = (
            None if test_seasons is None else frozenset(test_seasons)
        )

    def split(self, training_data: pl.DataFrame) -> Iterable[Fold]:
        """Yield one fold per gameweek after the training minimum.

        Gameweeks are ordered by ``(season, gw)``, which is
        chronological: season strings sort lexicographically in calendar
        order ("2025-26" < "2026-27") and gameweeks sort numerically
        within one. Each fold trains on every earlier gameweek and tests
        on the next.

        The ordinal is resolved once by joining the sorted distinct keys
        back onto the frame, so each fold is two comparisons rather than
        a membership test against a growing key list.

        Parameters
        ----------
        training_data : pl.DataFrame
            The model frame, carrying ``season`` and ``gw``.

        Yields
        ------
        Fold
            Folds in chronological order, carrying the same columns as
            ``training_data``. Nothing is yielded when there are not
            enough distinct gameweeks to form one fold, or when
            ``test_seasons`` matches none of them.
        """
        _ordinal_row_column = "_fold_ordinal"
        keys = (
            training_data.select("season", "gw")
            .unique()
            .sort(["season", "gw"])
            .with_row_index(_ordinal_row_column)
        )
        ordered = keys.rows()
        ordinals = training_data.join(keys, on=["season", "gw"], how="left")
        candidates = range(self.min_train_folds, len(ordered))
        testable = [
            index
            for index in candidates
            if self.test_seasons is None
            or ordered[index][1] in self.test_seasons
        ]
        if len(testable) != len(candidates):
            logger.info(
                "Cross-validating on %d of %d gameweek folds; dropped %d "
                "whose test gameweek falls outside %s.",
                len(testable),
                len(candidates),
                len(candidates) - len(testable),
                ", ".join(sorted(self.test_seasons or ())),
            )
        for index in testable:
            _, season, gw = ordered[index]
            yield Fold(
                train=ordinals.filter(
                    pl.col(_ordinal_row_column) < index
                ).drop(_ordinal_row_column),
                test=ordinals.filter(
                    pl.col(_ordinal_row_column) == index
                ).drop(_ordinal_row_column),
                test_key=FoldTestKey(season=season, gw=gw),
            )
