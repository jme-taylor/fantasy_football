from pathlib import Path
from unittest.mock import MagicMock

import polars as pl

from fantasy_football.extraction.player_match import (
    load_current_season_player_match,
)
from fantasy_football.fpl_types import FplPlayer
from fantasy_football.storage.database import get_connection, load_player_match


def _player(element_id: int) -> FplPlayer:
    return FplPlayer(
        id=element_id,
        first_name="A",
        second_name="B",
        web_name="AB",
        selected_by_percent=0.0,
        now_cost=50,
        team_id=1,
        element_type=3,
    )


def test_load_current_season_player_match_concatenates_and_upserts(
    tmp_path: Path,
) -> None:
    """Each player's history is concatenated, stamped, and written to the table."""
    fpl_api = MagicMock()
    fpl_api.get_players.return_value = [_player(5), _player(9)]

    def _history(element_id: int) -> pl.DataFrame:
        return pl.DataFrame(
            {
                "element": [element_id],
                "gw": [1],
                "opponent": [3],
                "is_home": [True],
                "minutes": [90],
                "total_points": [4],
            }
        )

    fpl_api.get_player_match_history.side_effect = _history

    connection = get_connection(tmp_path / "test.duckdb")
    try:
        load_current_season_player_match("2025-26", connection, fpl_api)
        out = load_player_match(connection)
    finally:
        connection.close()

    assert out.height == 2
    assert set(out["element"].to_list()) == {5, 9}
    assert out["season"].unique().to_list() == ["2025-26"]
