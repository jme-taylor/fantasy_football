import logging
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Protocol

import polars as pl

from fantasy_football.modelling.metrics import Metrics

logger = logging.getLogger(__name__)

DEFAULT_MIN_TRAIN_GAMEWEEKS = 10
DEFAULT_TEST_FRACTION = 0.2

# Prefix for the cross-validating strategies, so a mean of per-gameweek
# scores never lands in the same MLflow column as a single pooled score.
CV_PREFIX = "cv"

# Bookkeeping column carrying a row's chronological gameweek position.
# Always dropped before a fold is yielded.
ORDINAL_COLUMN = "_fold_ordinal"


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


@dataclass
class FoldResult:
    """One scored fold: its metrics and the rows they came from.

    The predictions are carried out of scoring rather than recomputed,
    so storing them costs nothing beyond the write.

    Attributes
    ----------
    metrics : Metrics
        The fold's scores.
    predictions : pl.DataFrame
        One row per scored row, shaped for the model's evaluation table
        but without ``run_id``, which only the run itself knows.
    """

    metrics: Metrics
    predictions: pl.DataFrame


def gameweek_ordinals(
    training_data: pl.DataFrame,
) -> tuple[list[tuple], pl.DataFrame]:
    """Return every ``(season, gw)`` key numbered chronologically.

    Season strings sort lexicographically in calendar order ("2025-26" <
    "2026-27") and gameweeks sort numerically within one, so sorting the
    distinct keys is enough to order them by time.

    Parameters
    ----------
    training_data : pl.DataFrame
        The model frame, carrying ``season`` and ``gw``.

    Returns
    -------
    tuple[list[tuple], pl.DataFrame]
        The sorted keys as ``(ordinal, season, gw)`` rows, and
        ``training_data`` with :data:`ORDINAL_COLUMN` joined on. Joining
        the ordinal once makes each fold two integer comparisons rather
        than a membership test against a growing key list.
    """
    keys = (
        training_data.select("season", "gw")
        .unique()
        .sort(["season", "gw"])
        .with_row_index(ORDINAL_COLUMN)
    )
    return keys.rows(), training_data.join(
        keys, on=["season", "gw"], how="left"
    )


class FoldStrategy(Protocol):
    """Protocol for a fold strategy.

    Any fold strategy must take a polars dataframe of training data spilt
    it into a a sequence of folds.
    """

    #: Prefix every metric this strategy's folds produce is logged under.
    #: None logs them under their bare names, which is right for a
    #: strategy whose scores need nothing to tell them apart.
    metric_prefix: str | None
    #: Whether a run reports the mean and spread across folds, or one
    #: fold's metrics directly. Fixed per strategy so the metric names a
    #: run logs never depend on how many folds the data allowed.
    aggregates: bool

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

    metric_prefix: str | None = CV_PREFIX
    aggregates = True

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

    metric_prefix: str | None = CV_PREFIX
    aggregates = True

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
        ordered, ordinals = gameweek_ordinals(training_data)
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
                train=ordinals.filter(pl.col(ORDINAL_COLUMN) < index).drop(
                    ORDINAL_COLUMN
                ),
                test=ordinals.filter(pl.col(ORDINAL_COLUMN) == index).drop(
                    ORDINAL_COLUMN
                ),
                test_key=FoldTestKey(season=season, gw=gw),
            )


class TrainTestSplitStrategy:
    """A single chronological holdout, expressed as a fraction of rows.

    Yields exactly one fold, so a train/test split is a single-fold
    cross validation and every caller of :class:`FoldStrategy` supports
    it unchanged.

    There is deliberately no season logic. The boundary is a percentage
    of the rows a fold may test on, so gameweeks of a new season join the
    frame and push the boundary forward without a code change.

    Parameters
    ----------
    test_fraction : float | None, optional
        Share of testable rows to hold out. Defaults to
        :data:`DEFAULT_TEST_FRACTION`.
    test_seasons : Sequence[str] | None, optional
        Seasons the holdout may be drawn from, and the population the
        fraction is measured against. Defaults to every season in the
        frame. Pass the seasons in which every model feature is actually
        published: taking the fraction over the whole frame would put the
        boundary years before coverage opens.

        Only the test side is restricted. The training side still spans
        every earlier row, uncovered seasons included, which is what the
        deployed model does.
    """

    metric_prefix: str | None = None
    aggregates = False

    def __init__(
        self,
        test_fraction: float | None = None,
        test_seasons: Sequence[str] | None = None,
    ) -> None:
        """Initialize the train/test split strategy.

        Parameters
        ----------
        test_fraction : float | None, optional
            Share of testable rows to hold out. Must lie strictly between
            zero and one, since either end leaves a side empty.
        test_seasons : Sequence[str] | None, optional
            Seasons the holdout may be drawn from. Pass None to use every
            season in the frame.
        """
        fraction = (
            DEFAULT_TEST_FRACTION if test_fraction is None else test_fraction
        )
        if not 0.0 < fraction < 1.0:
            raise ValueError(
                "test_fraction must be strictly between 0 and 1; "
                f"got {fraction}"
            )
        if test_seasons is not None and not test_seasons:
            raise ValueError(
                "test_seasons must name at least one season; pass None to "
                "hold out from every season in the frame"
            )
        self.test_fraction = fraction
        self.test_seasons = (
            None if test_seasons is None else frozenset(test_seasons)
        )

    def split(self, training_data: pl.DataFrame) -> Iterable[Fold]:
        """Yield one fold holding out the most recent gameweeks.

        The boundary snaps to a ``(season, gw)`` edge nearest the
        requested share of testable rows, so a fixture is never divided:
        two players in one match share their club's form columns and a
        correlated outcome, and splitting them would let the training
        side leak into the score.

        Parameters
        ----------
        training_data : pl.DataFrame
            The model frame, carrying ``season`` and ``gw``.

        Yields
        ------
        Fold
            One fold, carrying the same columns as ``training_data``.
            Nothing is yielded when either side would be empty, or when
            ``test_seasons`` matches no row.
        """
        ordered, ordinals = gameweek_ordinals(training_data)
        testable = (
            ordinals
            if self.test_seasons is None
            else ordinals.filter(
                pl.col("season").is_in(list(self.test_seasons))
            )
        )
        if testable.is_empty():
            return
        boundary = self._boundary(testable)
        if boundary is None:
            return
        _, season, gw = ordered[boundary]
        test = testable.filter(pl.col(ORDINAL_COLUMN) >= boundary)
        logger.info(
            "Holding out %d of %d testable rows (%.1f%%) from %s gw %d.",
            test.height,
            testable.height,
            100 * test.height / testable.height,
            season,
            gw,
        )
        yield Fold(
            train=ordinals.filter(pl.col(ORDINAL_COLUMN) < boundary).drop(
                ORDINAL_COLUMN
            ),
            test=test.drop(ORDINAL_COLUMN),
            test_key=FoldTestKey(season=season, gw=gw),
        )

    def _boundary(self, testable: pl.DataFrame) -> int | None:
        """Return the ordinal the holdout starts at, or None.

        Walks gameweeks backwards accumulating testable rows and stops
        once taking another would move further from the target, which is
        the nearest edge because the running total only grows.

        Parameters
        ----------
        testable : pl.DataFrame
            The rows a fold may test on, carrying
            :data:`ORDINAL_COLUMN`.

        Returns
        -------
        int | None
            The chosen ordinal, or None when every candidate would leave
            the training side empty.
        """
        counts = dict(testable.group_by(ORDINAL_COLUMN).len().rows())
        target = testable.height * self.test_fraction
        chosen: int | None = None
        best_gap = float("inf")
        cumulative = 0
        for ordinal in sorted(counts, reverse=True):
            cumulative += counts[ordinal]
            # Ordinal zero as the boundary would leave nothing to train on.
            if ordinal < 1:
                break
            gap = abs(cumulative - target)
            if gap >= best_gap:
                break
            chosen, best_gap = ordinal, gap
        return chosen
