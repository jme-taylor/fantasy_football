from pathlib import Path

import polars as pl  # noqa: F401  # used by later tasks appended to this file
import pytest  # noqa: F401  # used by later tasks appended to this file

from fantasy_football.storage.database import get_connection


def test_get_connection_creates_player_week_table(tmp_path: Path) -> None:
    """get_connection creates the DB file and an empty player_week table."""
    db_path = tmp_path / "test.duckdb"
    connection = get_connection(db_path)
    try:
        columns = [
            row[0]
            for row in connection.execute("DESCRIBE player_week").fetchall()
        ]
    finally:
        connection.close()

    assert db_path.exists()
    assert columns == [
        "season",
        "gw",
        "element",
        "name",
        "position",
        "team",
        "bonus",
        "minutes",
        "round",
        "total_points",
        "value",
    ]


def test_get_connection_is_idempotent(tmp_path: Path) -> None:
    """Opening the same DB twice does not error or duplicate the schema."""
    db_path = tmp_path / "test.duckdb"
    first = get_connection(db_path)
    first.close()
    second = get_connection(db_path)
    try:
        count = second.execute("SELECT COUNT(*) FROM player_week").fetchone()[0]
    finally:
        second.close()
    assert count == 0


from fantasy_football.storage.database import (  # noqa: E402
    PLAYER_WEEK_COLUMNS,
    coerce_player_week,
)


def test_coerce_player_week_normalises_gkp_and_selects_columns() -> None:
    """GKP collapses to GK; output is exactly PLAYER_WEEK_COLUMNS in order."""
    raw = pl.DataFrame(
        {
            "season": ["2020-21"],
            "gw": [1],
            "element": [1],
            "name": ["Player1"],
            "position": ["GKP"],
            "team": ["Arsenal"],
            "bonus": [1],
            "minutes": [90],
            "round": [1],
            "total_points": [6],
            "value": [50],
            "unused_extra": ["drop me"],
        }
    )

    result = coerce_player_week(raw)

    assert result.columns == PLAYER_WEEK_COLUMNS
    assert result["position"].to_list() == ["GK"]


def test_coerce_player_week_pins_dtypes() -> None:
    """Numeric columns arriving as Float64 are cast to Int64."""
    raw = pl.DataFrame(
        {
            "season": ["2020-21"],
            "gw": [1],
            "element": [1],
            "name": ["Player1"],
            "position": ["GK"],
            "team": ["Arsenal"],
            "bonus": [1.0],
            "minutes": [90.0],
            "round": [1],
            "total_points": [6],
            "value": [50],
        }
    )

    result = coerce_player_week(raw)

    assert result.schema["bonus"] == pl.Int64
    assert result.schema["minutes"] == pl.Int64
