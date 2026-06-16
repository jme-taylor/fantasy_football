from pathlib import Path

import duckdb
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
        count = second.execute("SELECT COUNT(*) FROM player_week").fetchone()[
            0
        ]
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


from fantasy_football.storage.database import (  # noqa: E402
    seasons_present,
    write_immutable_season,
)


def _historic_row(season: str, element: int, name: str) -> pl.DataFrame:
    """Build a one-row player-week frame for a given season/player."""
    return pl.DataFrame(
        {
            "season": [season],
            "gw": [1],
            "element": [element],
            "name": [name],
            "position": ["DEF"],
            "team": ["Arsenal"],
            "bonus": [1],
            "minutes": [90],
            "round": [1],
            "total_points": [6],
            "value": [50],
        }
    )


def test_seasons_present_reflects_writes(tmp_path: Path) -> None:
    """seasons_present returns the distinct seasons inserted so far."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        assert seasons_present(connection) == set()
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "P1"), "2020-21"
        )
        assert seasons_present(connection) == {"2020-21"}
    finally:
        connection.close()


def test_write_immutable_season_is_noop_when_present(tmp_path: Path) -> None:
    """Re-writing an existing season does not change or duplicate its rows."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "Original"), "2020-21"
        )
        # Attempt to overwrite the same season with different data.
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "Changed"), "2020-21"
        )
        rows = connection.execute(
            "SELECT name FROM player_week WHERE season = '2020-21'"
        ).fetchall()
    finally:
        connection.close()

    assert rows == [("Original",)]


from fantasy_football.storage.database import (  # noqa: E402
    upsert_current_season,
)


def _current_row(
    gw: int, element: int, total_points: int
) -> dict[str, object]:
    """Build one current-season player-week record as a dict of lists' values."""
    return {
        "season": "2025-26",
        "gw": gw,
        "element": element,
        "name": "Player",
        "position": "MID",
        "team": "Arsenal",
        "bonus": 0,
        "minutes": 90,
        "round": gw,
        "total_points": total_points,
        "value": 75,
    }


def _frame(records: list[dict[str, object]]) -> pl.DataFrame:
    """Build a player-week frame from a list of record dicts."""
    return pl.DataFrame(records)


def test_upsert_overwrites_changed_row_and_inserts_new_gw(
    tmp_path: Path,
) -> None:
    """A later upsert corrects an existing GW's value and adds the next GW."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        # First pull: GW1 provisional points.
        upsert_current_season(
            connection, _frame([_current_row(1, 1, 5)]), "2025-26"
        )
        # Second pull: GW1 corrected to 7, GW2 newly available.
        upsert_current_season(
            connection,
            _frame([_current_row(1, 1, 7), _current_row(2, 1, 9)]),
            "2025-26",
        )
        rows = connection.execute(
            "SELECT gw, total_points FROM player_week "
            "WHERE season = '2025-26' ORDER BY gw"
        ).fetchall()
    finally:
        connection.close()

    assert rows == [(1, 7), (2, 9)]


def test_upsert_does_not_touch_other_seasons(tmp_path: Path) -> None:
    """Upserting the current season leaves stored prior seasons intact."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "Old"), "2020-21"
        )
        upsert_current_season(
            connection, _frame([_current_row(1, 1, 5)]), "2025-26"
        )
        assert seasons_present(connection) == {"2020-21", "2025-26"}
    finally:
        connection.close()


from fantasy_football.storage import database  # noqa: E402
from fantasy_football.storage.database import (  # noqa: E402
    load_player_week,
    reset_database,
)


def test_load_player_week_returns_all_rows_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_player_week opens the default DB and returns every row, ordered."""
    db_path = tmp_path / "t.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    write_immutable_season(
        connection, _historic_row("2020-21", 2, "Older"), "2020-21"
    )
    upsert_current_season(
        connection, _frame([_current_row(1, 1, 5)]), "2025-26"
    )
    connection.close()

    result = load_player_week()

    assert result.columns == PLAYER_WEEK_COLUMNS
    assert result["season"].to_list() == ["2020-21", "2025-26"]


def test_reset_database_empties_the_table(tmp_path: Path) -> None:
    """reset_database drops all rows but leaves a usable empty table."""
    connection = get_connection(tmp_path / "t.duckdb")
    try:
        write_immutable_season(
            connection, _historic_row("2020-21", 1, "P1"), "2020-21"
        )
        reset_database(connection)
        assert seasons_present(connection) == set()
    finally:
        connection.close()


# ---------------------------------------------------------------------------
# team_fixture tests
# ---------------------------------------------------------------------------

from datetime import datetime  # noqa: E402

from fantasy_football.storage.database import (  # noqa: E402
    TEAM_FIXTURE_COLUMNS,
    coerce_team_fixture,
    load_team_fixture,
)


def _conn(tmp_path) -> duckdb.DuckDBPyConnection:
    return get_connection(tmp_path / "test.duckdb")


def _fixture_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2023-24", "2023-24"],
            "gw": [1, 1],
            "team": ["Arsenal", "Chelsea"],
            "is_home": [True, False],
            "opposition": ["Chelsea", "Arsenal"],
            "kickoff_time": [
                datetime(2023, 8, 11, 19, 0),
                datetime(2023, 8, 11, 19, 0),
            ],
        }
    )


def test_get_connection_creates_team_fixture_table(tmp_path) -> None:
    """get_connection creates the team_fixture table."""
    conn = _conn(tmp_path)
    try:
        columns = [
            row[0]
            for row in conn.execute("DESCRIBE team_fixture").fetchall()
        ]
    finally:
        conn.close()

    assert columns == TEAM_FIXTURE_COLUMNS


def test_coerce_team_fixture_selects_and_orders_columns() -> None:
    """coerce_team_fixture reduces a frame to the canonical columns/order."""
    frame = _fixture_frame().with_columns(pl.lit("extra").alias("junk"))
    shaped = coerce_team_fixture(frame)
    assert shaped.columns == TEAM_FIXTURE_COLUMNS


def test_load_team_fixture_round_trips(tmp_path) -> None:
    """A frame inserted directly is read back via load_team_fixture."""
    conn = _conn(tmp_path)
    try:
        conn.register("incoming", coerce_team_fixture(_fixture_frame()).to_arrow())
        conn.execute("INSERT INTO team_fixture SELECT * FROM incoming")
        conn.unregister("incoming")
        out = load_team_fixture(conn)
        assert out.columns == TEAM_FIXTURE_COLUMNS
        assert out.height == 2
        assert set(out["team"].to_list()) == {"Arsenal", "Chelsea"}
    finally:
        conn.close()


def test_reset_database_drops_team_fixture_rows(tmp_path) -> None:
    """reset_database recreates an empty team_fixture table."""
    conn = _conn(tmp_path)
    try:
        conn.register("incoming", coerce_team_fixture(_fixture_frame()).to_arrow())
        conn.execute("INSERT INTO team_fixture SELECT * FROM incoming")
        conn.unregister("incoming")
        reset_database(conn)
        assert load_team_fixture(conn).height == 0
    finally:
        conn.close()


from fantasy_football.storage.database import (  # noqa: E402
    fixture_seasons_present,
    upsert_current_fixtures,
    write_immutable_fixtures,
)


def test_write_immutable_fixtures_inserts_once(tmp_path) -> None:
    """write_immutable_fixtures inserts a new season but skips a present one."""
    conn = _conn(tmp_path)
    try:
        write_immutable_fixtures(conn, _fixture_frame(), "2023-24")
        assert load_team_fixture(conn).height == 2
        # Second call with different data is ignored — season already present.
        changed = _fixture_frame().with_columns(pl.lit("Spurs").alias("team"))
        write_immutable_fixtures(conn, changed, "2023-24")
        teams = set(load_team_fixture(conn)["team"].to_list())
        assert teams == {"Arsenal", "Chelsea"}
    finally:
        conn.close()


def test_upsert_current_fixtures_replaces_season(tmp_path) -> None:
    """upsert_current_fixtures deletes then reinserts the season's rows."""
    conn = _conn(tmp_path)
    try:
        upsert_current_fixtures(conn, _fixture_frame(), "2023-24")
        replacement = pl.DataFrame(
            {
                "season": ["2023-24"],
                "gw": [2],
                "team": ["Arsenal"],
                "is_home": [True],
                "opposition": ["Spurs"],
                "kickoff_time": [datetime(2023, 8, 19, 15, 0)],
            }
        )
        upsert_current_fixtures(conn, replacement, "2023-24")
        out = load_team_fixture(conn)
        assert out.height == 1
        assert out.row(0, named=True)["opposition"] == "Spurs"
    finally:
        conn.close()


def test_fixture_seasons_present(tmp_path) -> None:
    """fixture_seasons_present returns distinct stored seasons."""
    conn = _conn(tmp_path)
    try:
        assert fixture_seasons_present(conn) == set()
        write_immutable_fixtures(conn, _fixture_frame(), "2023-24")
        assert fixture_seasons_present(conn) == {"2023-24"}
    finally:
        conn.close()
