"""Tests for the cross-source overlap validation."""

import polars as pl

from fantasy_football.storage.validation import (
    compare_overlap,
    validate_overlap,
)


def _fpl_frame() -> pl.DataFrame:
    """Return two Vaastav player-gameweek rows."""
    return pl.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "gw": [1, 1],
            "element": [10, 11],
            "minutes": [90, 45],
            "goals_scored": [1, 0],
            "assists": [0, 1],
            "saves": [0, 0],
            "penalties_missed": [0, 0],
        }
    )


def _opta_frame(goals: list[int]) -> pl.DataFrame:
    """Return two FCI prem rows with the given goal counts."""
    return pl.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "gw": [1, 1],
            "element": [10, 11],
            "competition": ["prem", "prem"],
            "minutes_played": [90, 45],
            "goals": goals,
            "assists": [0, 1],
            "saves": [0, 0],
            "penalties_missed": [0, 0],
        }
    )


def test_agreeing_sources_report_no_disagreements():
    """Identical stats across sources yield a zero disagreement rate."""
    result = compare_overlap(_fpl_frame(), _opta_frame([1, 0]))
    goals = result.filter(pl.col("column") == "goals_scored")
    assert goals["disagreements"].item() == 0
    assert goals["disagreement_rate"].item() == 0.0


def test_disagreeing_sources_are_counted():
    """A mismatched value is counted and rated."""
    result = compare_overlap(_fpl_frame(), _opta_frame([9, 0]))
    goals = result.filter(pl.col("column") == "goals_scored")
    assert goals["disagreements"].item() == 1
    assert goals["disagreement_rate"].item() == 0.5


def test_non_prem_rows_are_excluded():
    """European minutes must not count towards the FPL comparison."""
    opta = pl.concat(
        [
            _opta_frame([1, 0]),
            _opta_frame([3, 3]).with_columns(
                pl.lit("champions").alias("competition")
            ),
        ]
    )
    result = compare_overlap(_fpl_frame(), opta)
    goals = result.filter(pl.col("column") == "goals_scored")
    assert goals["disagreements"].item() == 0


def test_double_gameweeks_are_summed_before_comparing():
    """Both legs of a double gameweek sum to one player-gameweek total."""
    fpl = _fpl_frame().head(1).with_columns(pl.lit(2).alias("goals_scored"))
    opta = pl.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "gw": [1, 1],
            "element": [10, 10],
            "competition": ["prem", "prem"],
            "minutes_played": [90, 90],
            "goals": [1, 1],
            "assists": [0, 0],
            "saves": [0, 0],
            "penalties_missed": [0, 0],
        }
    )
    result = compare_overlap(fpl, opta)
    goals = result.filter(pl.col("column") == "goals_scored")
    assert goals["disagreements"].item() == 0


def test_seasons_in_only_one_source_are_ignored():
    """A season only Vaastav has contributes nothing to the comparison."""
    fpl = pl.concat(
        [
            _fpl_frame(),
            _fpl_frame().with_columns(pl.lit("2016-17").alias("season")),
        ]
    )
    result = compare_overlap(fpl, _opta_frame([1, 0]))
    assert result["compared"].max() == 2


def test_validate_overlap_on_empty_tables_returns_zero_rows_compared(db):
    """With nothing loaded, validation reports no comparisons, not a crash."""
    result = validate_overlap(db)
    assert result["compared"].to_list() == [0] * result.height
