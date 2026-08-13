import polars as pl
import pytest

from fantasy_football.modelling.folds import (
    DEFAULT_MIN_TRAIN_GAMEWEEKS,
    DEFAULT_TEST_FRACTION,
    ExpandingGameweekFoldStrategy,
    SeasonFoldStrategy,
    TrainTestSplitStrategy,
)


def _two_season_frame() -> pl.DataFrame:
    """Three gameweeks in an uncovered season, three in a covered one."""
    return pl.DataFrame(
        {
            "season": ["2024-25"] * 3 + ["2025-26"] * 3,
            "gw": [1, 2, 3, 1, 2, 3],
        }
    )


def test_season_folds_expand_over_sorted_seasons():
    """Folds expand across seasons regardless of input row order."""
    frame = pl.DataFrame(
        {"season": ["2026-27", "2024-25", "2025-26"], "gw": [1, 1, 1]}
    )
    folds = list(SeasonFoldStrategy().split(frame))
    assert [fold.test_key.season for fold in folds] == ["2025-26", "2026-27"]


def test_season_folds_test_the_whole_season():
    """A season fold names no gameweek, because it holds out all of them."""
    folds = list(SeasonFoldStrategy().split(_two_season_frame()))
    assert [fold.test_key.gw for fold in folds] == [None]


def test_season_folds_train_on_every_earlier_season():
    """Train holds the prior seasons; test holds only the test season."""
    frame = pl.DataFrame(
        {
            "season": ["2023-24", "2024-25", "2024-25", "2025-26"],
            "gw": [1, 1, 2, 1],
        }
    )
    folds = list(SeasonFoldStrategy().split(frame))
    last = folds[-1]
    assert last.train["season"].unique().sort().to_list() == [
        "2023-24",
        "2024-25",
    ]
    assert last.train.height == 3
    assert last.test["season"].unique().to_list() == ["2025-26"]


def test_season_folds_empty_with_a_single_season():
    """One season cannot be split into a train and a test side."""
    frame = pl.DataFrame({"season": ["2025-26"] * 3, "gw": [1, 2, 3]})
    assert list(SeasonFoldStrategy().split(frame)) == []


def test_expanding_strategy_tests_every_season_by_default():
    """Without test_seasons, every gameweek after the minimum is a fold."""
    strategy = ExpandingGameweekFoldStrategy(min_train_folds=1)
    folds = list(strategy.split(_two_season_frame()))
    assert [(fold.test_key.season, fold.test_key.gw) for fold in folds] == [
        ("2024-25", 2),
        ("2024-25", 3),
        ("2025-26", 1),
        ("2025-26", 2),
        ("2025-26", 3),
    ]


def test_expanding_strategy_only_tests_named_seasons():
    """test_seasons drops folds whose test gameweek predates coverage."""
    strategy = ExpandingGameweekFoldStrategy(
        min_train_folds=1, test_seasons=["2025-26"]
    )
    folds = list(strategy.split(_two_season_frame()))
    assert [(fold.test_key.season, fold.test_key.gw) for fold in folds] == [
        ("2025-26", 1),
        ("2025-26", 2),
        ("2025-26", 3),
    ]


def test_expanding_strategy_still_trains_on_unnamed_seasons():
    """Only the test side is restricted; training keeps earlier seasons."""
    strategy = ExpandingGameweekFoldStrategy(
        min_train_folds=1, test_seasons=["2025-26"]
    )
    first = next(iter(strategy.split(_two_season_frame())))
    assert first.train["season"].unique().to_list() == ["2024-25"]
    assert first.train.height == 3


def test_expanding_strategy_yields_nothing_when_no_season_is_named():
    """A test_seasons set matching no fold yields no folds at all."""
    strategy = ExpandingGameweekFoldStrategy(
        min_train_folds=1, test_seasons=["2026-27"]
    )
    assert list(strategy.split(_two_season_frame())) == []


def test_expanding_strategy_rejects_empty_test_seasons():
    """An empty test_seasons is a wiring mistake, not "test everything"."""
    with pytest.raises(ValueError):
        ExpandingGameweekFoldStrategy(test_seasons=[])


def test_expanding_strategy_rejects_min_train_folds_below_one():
    """A fold has to train on something, so zero gameweeks is a mistake."""
    with pytest.raises(ValueError):
        ExpandingGameweekFoldStrategy(min_train_folds=0)


def test_expanding_strategy_defaults_min_train_folds():
    """Omitting min_train_folds falls back to the module default."""
    strategy = ExpandingGameweekFoldStrategy()
    assert strategy.min_train_folds == DEFAULT_MIN_TRAIN_GAMEWEEKS


def test_expanding_strategy_empty_when_too_few_gameweeks():
    """Fewer gameweeks than the training minimum yields no folds."""
    strategy = ExpandingGameweekFoldStrategy()
    assert list(strategy.split(_two_season_frame())) == []


def test_expanding_strategy_keeps_rows_sharing_a_gameweek_together():
    """A gameweek is one fold, however many rows it holds."""
    frame = pl.DataFrame(
        {"season": ["2025-26"] * 3, "gw": [1, 1, 2], "player_id": [1, 2, 1]}
    )
    strategy = ExpandingGameweekFoldStrategy(min_train_folds=1)
    folds = list(strategy.split(frame))
    assert len(folds) == 1
    assert folds[0].train.height == 2
    assert folds[0].test.height == 1


def test_expanding_strategy_preserves_columns_and_partitions_rows():
    """Folds carry the input's columns, no bookkeeping ones, no leakage."""
    frame = _two_season_frame().with_columns(points=pl.lit(1.0))
    strategy = ExpandingGameweekFoldStrategy(min_train_folds=1)
    for fold in strategy.split(frame):
        assert fold.train.columns == frame.columns
        assert fold.test.columns == frame.columns
        assert fold.test["gw"].unique().to_list() == [fold.test_key.gw]
        overlap = fold.train.join(fold.test, on=["season", "gw"], how="semi")
        assert overlap.height == 0


# --- Train/test split -------------------------------------------------


def _even_frame(n_gws: int = 10, rows_per_gw: int = 10) -> pl.DataFrame:
    """One season of equally sized gameweeks."""
    return pl.DataFrame(
        {
            "season": ["2025-26"] * (n_gws * rows_per_gw),
            "gw": [
                gw for gw in range(1, n_gws + 1) for _ in range(rows_per_gw)
            ],
        }
    )


def test_split_holds_out_the_requested_fraction():
    """A fifth of the rows land on the test side."""
    folds = list(
        TrainTestSplitStrategy(test_fraction=0.2).split(_even_frame())
    )
    assert folds[0].train.height == 80
    assert folds[0].test.height == 20


def test_split_yields_exactly_one_fold():
    """A train/test split is a single-fold cross validation."""
    assert len(list(TrainTestSplitStrategy().split(_even_frame()))) == 1


def test_split_test_side_is_the_most_recent_gameweeks():
    """The holdout is the chronological tail, never an earlier slice."""
    fold = next(
        iter(TrainTestSplitStrategy(test_fraction=0.2).split(_even_frame()))
    )
    assert fold.test["gw"].unique().sort().to_list() == [9, 10]
    assert fold.train["gw"].max() == 8


def test_split_names_the_first_test_gameweek():
    """The fold key points at where the holdout starts."""
    fold = next(
        iter(TrainTestSplitStrategy(test_fraction=0.2).split(_even_frame()))
    )
    assert (fold.test_key.season, fold.test_key.gw) == ("2025-26", 9)


def test_split_snaps_to_a_gameweek_edge():
    """No gameweek is ever divided across the boundary."""
    frame = pl.DataFrame(
        {
            "season": ["2025-26"] * 9,
            "gw": [1, 1, 1, 2, 2, 3, 3, 3, 3],
        }
    )
    fold = next(iter(TrainTestSplitStrategy(test_fraction=0.3).split(frame)))
    overlap = fold.train.join(fold.test, on=["season", "gw"], how="semi")
    assert overlap.height == 0


def test_split_orders_gameweeks_across_seasons():
    """Season then gameweek is chronological, so the tail spans seasons."""
    frame = pl.DataFrame(
        {
            "season": ["2024-25"] * 4 + ["2025-26"] * 4,
            "gw": [1, 2, 3, 4, 1, 2, 3, 4],
        }
    )
    fold = next(iter(TrainTestSplitStrategy(test_fraction=0.25).split(frame)))
    assert fold.test["season"].unique().to_list() == ["2025-26"]
    assert fold.test["gw"].unique().sort().to_list() == [3, 4]


def test_split_takes_the_fraction_of_testable_rows_only():
    """The denominator is the seasons a fold may test on, not the frame."""
    frame = pl.DataFrame(
        {
            "season": ["2024-25"] * 20 + ["2025-26"] * 10,
            "gw": list(range(1, 21)) + list(range(1, 11)),
        }
    )
    strategy = TrainTestSplitStrategy(
        test_fraction=0.2, test_seasons=["2025-26"]
    )
    fold = next(iter(strategy.split(frame)))
    assert fold.test.height == 2
    assert fold.test["season"].unique().to_list() == ["2025-26"]


def test_split_still_trains_on_unnamed_seasons():
    """Only the test side is restricted; training keeps earlier seasons."""
    strategy = TrainTestSplitStrategy(
        test_fraction=0.4, test_seasons=["2025-26"]
    )
    fold = next(iter(strategy.split(_two_season_frame())))
    assert fold.train["season"].unique().sort().to_list() == [
        "2024-25",
        "2025-26",
    ]


def test_split_empty_when_a_side_would_be_empty():
    """One gameweek cannot be divided into a train and a test side."""
    frame = pl.DataFrame({"season": ["2025-26"] * 3, "gw": [1, 1, 1]})
    assert list(TrainTestSplitStrategy().split(frame)) == []


def test_split_empty_when_no_season_is_testable():
    """A test_seasons set matching no row yields no fold at all."""
    strategy = TrainTestSplitStrategy(test_seasons=["2026-27"])
    assert list(strategy.split(_two_season_frame())) == []


def test_split_rejects_a_fraction_outside_the_unit_interval():
    """A fraction of none or all of the rows leaves a side empty."""
    with pytest.raises(ValueError):
        TrainTestSplitStrategy(test_fraction=0.0)
    with pytest.raises(ValueError):
        TrainTestSplitStrategy(test_fraction=1.0)


def test_split_rejects_empty_test_seasons():
    """An empty test_seasons is a wiring mistake, not "test everything"."""
    with pytest.raises(ValueError):
        TrainTestSplitStrategy(test_seasons=[])


def test_split_defaults_the_fraction():
    """Omitting the fraction falls back to the module default."""
    assert TrainTestSplitStrategy().test_fraction == DEFAULT_TEST_FRACTION


def test_split_preserves_columns():
    """Folds carry the input's columns and no bookkeeping ones."""
    frame = _even_frame().with_columns(points=pl.lit(1.0))
    fold = next(iter(TrainTestSplitStrategy().split(frame)))
    assert fold.train.columns == frame.columns
    assert fold.test.columns == frame.columns


def test_strategies_name_themselves_for_metric_prefixing():
    """Holdout and cross-validated metrics must not share a name."""
    assert TrainTestSplitStrategy().metric_prefix == "holdout"
    assert SeasonFoldStrategy().metric_prefix == "cv"
    assert ExpandingGameweekFoldStrategy().metric_prefix == "cv"
