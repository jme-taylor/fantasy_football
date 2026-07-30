from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from fantasy_football.extraction.availability import (
    load_player_availability_data,
)
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import PLAYER_AVAILABILITY


def _frame(season: str, chance: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [season],
            "gw": [1],
            "element": [10],
            "chance_of_playing_this_round": [chance],
        }
    )


def test_load_player_availability_data_backfills_and_upserts(
    tmp_path: Path,
) -> None:
    """Each season in range is built once and written; current season upserts."""
    extractor = MagicMock()
    extractor.build_player_chance_of_playing.side_effect = (
        lambda season: _frame(season, 75)
    )

    connection = get_connection(tmp_path / "test.duckdb")
    try:
        load_player_availability_data(
            connection, current_season="2023-24", extractor=extractor
        )
        out = PLAYER_AVAILABILITY.load(connection)
    finally:
        connection.close()

    # FPLCACHE_FIRST_SEASON (2022-23) through 2023-24 inclusive.
    assert set(out["season"].to_list()) == {"2022-23", "2023-24"}


def test_load_player_availability_data_skips_present_immutable_seasons(
    tmp_path: Path,
) -> None:
    """A completed season already stored is not re-fetched."""
    extractor = MagicMock()
    extractor.build_player_chance_of_playing.side_effect = (
        lambda season: _frame(season, 75)
    )

    connection = get_connection(tmp_path / "test.duckdb")
    try:
        load_player_availability_data(
            connection, current_season="2023-24", extractor=extractor
        )
        first_call_count = extractor.build_player_chance_of_playing.call_count
        # Second run: 2022-23 is immutable+present, 2023-24 is current.
        load_player_availability_data(
            connection, current_season="2023-24", extractor=extractor
        )
    finally:
        connection.close()

    built_seasons = [
        call.args[0]
        for call in extractor.build_player_chance_of_playing.call_args_list[
            first_call_count:
        ]
    ]
    assert built_seasons == ["2023-24"]
