import pytest

from fantasy_football.extraction.seasons import (
    DataSource,
    season_long_to_short,
    season_short_to_long,
    seasons_in_range,
    source_for_season,
)


@pytest.mark.parametrize(
    "short, long",
    [
        ("2025-26", "2025-2026"),
        ("2024-25", "2024-2025"),
        ("2016-17", "2016-2017"),
    ],
)
def test_short_to_long_roundtrip(short: str, long: str) -> None:
    """Verify short-to-long and long-to-short conversions are inverses."""
    assert season_short_to_long(short) == long
    assert season_long_to_short(long) == short


def test_short_to_long_handles_century_rollover() -> None:
    """Verify 1999-00 correctly expands to 1999-2000."""
    assert season_short_to_long("1999-00") == "1999-2000"


def test_source_for_season_vaastav_for_historic() -> None:
    """Verify seasons up to and including 2024-25 route to Vaastav."""
    assert source_for_season("2024-25") == DataSource.VAASTAV


def test_source_for_season_fci_for_current() -> None:
    """Verify the current 2025-26 season routes to FCI."""
    assert source_for_season("2025-26") == DataSource.FCI


def test_source_for_season_fci_for_future() -> None:
    """Verify future seasons beyond 2025-26 also route to FCI."""
    assert source_for_season("2026-27") == DataSource.FCI


def test_seasons_in_range_is_inclusive() -> None:
    """Verify seasons_in_range returns all seasons inclusive of first and last."""
    assert seasons_in_range("2022-23", "2025-26") == [
        "2022-23",
        "2023-24",
        "2024-25",
        "2025-26",
    ]


def test_seasons_in_range_single_season() -> None:
    """Verify seasons_in_range returns a single-element list when first==last."""
    assert seasons_in_range("2025-26", "2025-26") == ["2025-26"]


def test_seasons_in_range_handles_century_rollover() -> None:
    """Verify seasons_in_range handles the 1999-00 to 2000-01 century rollover."""
    assert seasons_in_range("1999-00", "2000-01") == ["1999-00", "2000-01"]
