import pytest

from fantasy_football.extraction.seasons import (
    DataSource,
    season_long_to_short,
    season_short_to_long,
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
