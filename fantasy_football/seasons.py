"""Season-string conversions and data-source routing.

Three season-string formats are in play across the project:

* short form, e.g. ``"2025-26"`` — local folders and ``CURRENT_SEASON``;
* long form, e.g. ``"2025-2026"`` — FCI repo paths and the cutoff constants;
* Vaastav repo paths use the short form.
"""

from enum import Enum

from fantasy_football.constants import VASTAAV_LAST_SEASON


class DataSource(Enum):
    """Which upstream repository a season's data comes from."""

    VAASTAV = "vaastav"
    FCI = "fci"


def season_short_to_long(season: str) -> str:
    """Convert a short season string to long form (``2025-26`` -> ``2025-2026``)."""
    start, end = season.split("-")
    century = start[:2]
    # Handle century rollover, e.g. 1999-00 -> 1999-2000.
    if end == "00":
        century = str(int(century) + 1)
    return f"{start}-{century}{end}"


def season_long_to_short(season: str) -> str:
    """Convert a long season string to short form (``2025-2026`` -> ``2025-26``)."""
    start, end = season.split("-")
    return f"{start}-{end[2:]}"


def source_for_season(season: str) -> DataSource:
    """Return the data source for a short-form season string.

    Seasons after Vaastav's last actively-updated season come from FCI.
    """
    last_vaastav_short = season_long_to_short(VASTAAV_LAST_SEASON)
    return (
        DataSource.VAASTAV if season <= last_vaastav_short else DataSource.FCI
    )
