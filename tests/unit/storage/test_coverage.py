"""Tests for the per-column season coverage maps."""

import pytest

from fantasy_football.storage.coverage import (
    FCI_COLUMN_SEASONS,
    FCI_SEASONS,
    FPL_STAT_SEASONS,
    VAASTAV_COLUMN_SEASONS,
    seasons_covering,
)
from fantasy_football.storage.tables import PLAYER_MATCH_FPL, PLAYER_MATCH_OPTA


def test_every_vaastav_schema_column_has_coverage():
    """No stored Vaastav column is missing from the coverage map."""
    assert set(PLAYER_MATCH_FPL.columns) == set(VAASTAV_COLUMN_SEASONS)


def test_every_fci_schema_column_has_coverage():
    """No stored FCI column is missing from the coverage map."""
    assert set(PLAYER_MATCH_OPTA.columns) == set(FCI_COLUMN_SEASONS)


def test_coverage_seasons_are_known_seasons():
    """Coverage maps only reference seasons the source actually has."""
    for seasons in VAASTAV_COLUMN_SEASONS.values():
        assert set(seasons) <= set(FPL_STAT_SEASONS)
    for seasons in FCI_COLUMN_SEASONS.values():
        assert set(seasons) <= set(FCI_SEASONS)


def test_expected_goals_starts_in_2022_23():
    """expected_goals is a 2022-23-onwards column, not a full-history one."""
    assert VAASTAV_COLUMN_SEASONS["expected_goals"] == (
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    )


def test_corners_is_2026_27_only():
    """FCI added corners in 2026-27."""
    assert FCI_COLUMN_SEASONS["corners"] == ("2026-27",)


def test_seasons_covering_intersects():
    """The intersection of two columns' coverage is returned, sorted."""
    result = seasons_covering(
        VAASTAV_COLUMN_SEASONS, ["expected_goals", "tackles"]
    )
    assert result == ("2025-26",)


def test_seasons_covering_single_column():
    """A single column returns its own coverage."""
    assert seasons_covering(VAASTAV_COLUMN_SEASONS, ["starts"]) == (
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    )


def test_seasons_covering_no_columns_returns_all_seasons():
    """An empty column list is covered by every season in the map."""
    assert seasons_covering(VAASTAV_COLUMN_SEASONS, []) == FPL_STAT_SEASONS


def test_seasons_covering_rejects_unknown_column():
    """An unknown column is a programming error, not an empty result."""
    with pytest.raises(KeyError):
        seasons_covering(VAASTAV_COLUMN_SEASONS, ["not_a_column"])
