from datetime import date, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from fantasy_football.storage.database import (
    check_schema_drift,
    get_connection,
)
from fantasy_football.storage.tables import (
    MINUTES_PREDICTION,
    PLAYER_AVAILABILITY,
    PLAYER_MATCH,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)


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
        row = second.execute("SELECT COUNT(*) FROM player_week").fetchone()
    finally:
        second.close()
    assert row is not None
    count = row[0]
    assert count == 0


def test_get_connection_raises_on_schema_drift(tmp_path: Path) -> None:
    """A stored table missing a declared column fails loudly, not obscurely.

    CREATE TABLE IF NOT EXISTS cannot add a column to an existing table, so
    a spec change against an old database file would otherwise surface as a
    DuckDB Binder Error on the first read.
    """
    db_path = tmp_path / "drift.duckdb"
    connection = get_connection(db_path)
    connection.execute(
        "ALTER TABLE minutes_prediction DROP COLUMN snapshot_captured_at"
    )
    connection.close()

    with pytest.raises(RuntimeError) as excinfo:
        get_connection(db_path).close()

    message = str(excinfo.value)
    assert "minutes_prediction" in message
    assert "snapshot_captured_at" in message
    assert "rebuild=True" in message


def test_get_connection_skips_the_drift_check_for_a_rebuild(
    tmp_path: Path,
) -> None:
    """A rebuild must be able to open a drifted database in order to fix it.

    ``main(rebuild=True)`` opens a connection *before* dropping the tables,
    so an unconditional guard would make a drifted database impossible to
    rebuild -- the very remedy the error message recommends.
    """
    db_path = tmp_path / "drift.duckdb"
    connection = get_connection(db_path)
    connection.execute(
        "ALTER TABLE minutes_prediction DROP COLUMN snapshot_captured_at"
    )
    connection.close()

    connection = get_connection(db_path, check_drift=False)
    try:
        reset_database(connection)
    finally:
        connection.close()

    # After the rebuild the guard is satisfied again.
    get_connection(db_path).close()


def test_check_schema_drift_passes_on_a_fresh_database(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Every declared column exists after a normal create, so nothing raises."""
    check_schema_drift(db)


def test_coerce_player_week_normalises_gkp_and_selects_columns() -> None:
    """GKP collapses to GK; output is exactly PLAYER_WEEK.columns in order."""
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

    result = PLAYER_WEEK.coerce(raw)

    assert result.columns == PLAYER_WEEK.columns
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

    result = PLAYER_WEEK.coerce(raw)

    assert result.schema["bonus"] == pl.Int64
    assert result.schema["minutes"] == pl.Int64


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


def test_seasons_present_reflects_writes(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """seasons_present returns the distinct seasons inserted so far."""
    connection = db
    assert PLAYER_WEEK.seasons_present(connection) == set()
    PLAYER_WEEK.write_immutable(
        connection, _historic_row("2020-21", 1, "P1"), "2020-21"
    )
    assert PLAYER_WEEK.seasons_present(connection) == {"2020-21"}


def test_write_immutable_season_is_noop_when_present(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Re-writing an existing season does not change or duplicate its rows."""
    connection = db
    PLAYER_WEEK.write_immutable(
        connection, _historic_row("2020-21", 1, "Original"), "2020-21"
    )
    # Attempt to overwrite the same season with different data.
    PLAYER_WEEK.write_immutable(
        connection, _historic_row("2020-21", 1, "Changed"), "2020-21"
    )
    rows = connection.execute(
        "SELECT name FROM player_week WHERE season = '2020-21'"
    ).fetchall()

    assert rows == [("Original",)]


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
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A later upsert corrects an existing GW's value and adds the next GW."""
    connection = db
    # First pull: GW1 provisional points.
    PLAYER_WEEK.upsert_current(
        connection, _frame([_current_row(1, 1, 5)]), "2025-26"
    )
    # Second pull: GW1 corrected to 7, GW2 newly available.
    PLAYER_WEEK.upsert_current(
        connection,
        _frame([_current_row(1, 1, 7), _current_row(2, 1, 9)]),
        "2025-26",
    )
    rows = connection.execute(
        "SELECT gw, total_points FROM player_week "
        "WHERE season = '2025-26' ORDER BY gw"
    ).fetchall()

    assert rows == [(1, 7), (2, 9)]


def test_upsert_does_not_touch_other_seasons(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Upserting the current season leaves stored prior seasons intact."""
    connection = db
    PLAYER_WEEK.write_immutable(
        connection, _historic_row("2020-21", 1, "Old"), "2020-21"
    )
    PLAYER_WEEK.upsert_current(
        connection, _frame([_current_row(1, 1, 5)]), "2025-26"
    )
    assert PLAYER_WEEK.seasons_present(connection) == {
        "2020-21",
        "2025-26",
    }


from fantasy_football.storage import database  # noqa: E402
from fantasy_football.storage.database import reset_database  # noqa: E402


def test_load_player_week_returns_all_rows_ordered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """PLAYER_WEEK.load opens the default DB and returns every row, ordered."""
    db_path = tmp_path / "t.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    connection = get_connection(db_path)
    PLAYER_WEEK.write_immutable(
        connection, _historic_row("2020-21", 2, "Older"), "2020-21"
    )
    PLAYER_WEEK.upsert_current(
        connection, _frame([_current_row(1, 1, 5)]), "2025-26"
    )
    connection.close()

    result = PLAYER_WEEK.load()

    assert result.columns == PLAYER_WEEK.columns
    assert result["season"].to_list() == ["2020-21", "2025-26"]


def test_reset_database_empties_the_table(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """reset_database drops all rows but leaves a usable empty table."""
    connection = db
    PLAYER_WEEK.write_immutable(
        connection, _historic_row("2020-21", 1, "P1"), "2020-21"
    )
    reset_database(connection)
    assert PLAYER_WEEK.seasons_present(connection) == set()


# ---------------------------------------------------------------------------
# team_fixture tests
# ---------------------------------------------------------------------------


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


def test_get_connection_creates_team_fixture_table(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """get_connection creates the team_fixture table."""
    columns = [
        row[0] for row in db.execute("DESCRIBE team_fixture").fetchall()
    ]

    assert columns == TEAM_FIXTURE.columns


def test_coerce_team_fixture_selects_and_orders_columns() -> None:
    """TEAM_FIXTURE.coerce reduces a frame to the canonical columns/order."""
    frame = _fixture_frame().with_columns(pl.lit("extra").alias("junk"))
    shaped = TEAM_FIXTURE.coerce(frame)
    assert shaped.columns == TEAM_FIXTURE.columns


def test_load_team_fixture_round_trips(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A frame inserted directly is read back via TEAM_FIXTURE.load."""
    conn = db
    conn.register("incoming", TEAM_FIXTURE.coerce(_fixture_frame()).to_arrow())
    conn.execute("INSERT INTO team_fixture SELECT * FROM incoming")
    conn.unregister("incoming")
    out = TEAM_FIXTURE.load(conn)
    assert out.columns == TEAM_FIXTURE.columns
    assert out.height == 2
    assert set(out["team"].to_list()) == {"Arsenal", "Chelsea"}


def test_reset_database_drops_team_fixture_rows(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """reset_database recreates an empty team_fixture table."""
    conn = db
    conn.register("incoming", TEAM_FIXTURE.coerce(_fixture_frame()).to_arrow())
    conn.execute("INSERT INTO team_fixture SELECT * FROM incoming")
    conn.unregister("incoming")
    reset_database(conn)
    assert TEAM_FIXTURE.load(conn).height == 0


def test_write_immutable_fixtures_inserts_once(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """TEAM_FIXTURE.write_immutable inserts a new season but skips a present one."""
    conn = db
    TEAM_FIXTURE.write_immutable(conn, _fixture_frame(), "2023-24")
    assert TEAM_FIXTURE.load(conn).height == 2
    # Second call with different data is ignored — season already present.
    changed = _fixture_frame().with_columns(pl.lit("Spurs").alias("team"))
    TEAM_FIXTURE.write_immutable(conn, changed, "2023-24")
    teams = set(TEAM_FIXTURE.load(conn)["team"].to_list())
    assert teams == {"Arsenal", "Chelsea"}


def test_upsert_current_fixtures_replaces_season(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """TEAM_FIXTURE.upsert_current deletes then reinserts the season's rows."""
    conn = db
    TEAM_FIXTURE.upsert_current(conn, _fixture_frame(), "2023-24")
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
    TEAM_FIXTURE.upsert_current(conn, replacement, "2023-24")
    out = TEAM_FIXTURE.load(conn)
    assert out.height == 1
    assert out.row(0, named=True)["opposition"] == "Spurs"


def test_fixture_seasons_present(db: duckdb.DuckDBPyConnection) -> None:
    """TEAM_FIXTURE.seasons_present returns distinct stored seasons."""
    conn = db
    assert TEAM_FIXTURE.seasons_present(conn) == set()
    TEAM_FIXTURE.write_immutable(conn, _fixture_frame(), "2023-24")
    assert TEAM_FIXTURE.seasons_present(conn) == {"2023-24"}


def _player_match_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema_overrides={"is_home": pl.Boolean})


def test_get_connection_creates_player_match_table(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """get_connection creates an empty player_match table with the right columns."""
    connection = db
    columns = [
        row[0]
        for row in connection.execute("DESCRIBE player_match").fetchall()
    ]
    assert columns == PLAYER_MATCH.columns


def test_coerce_player_match_selects_and_pins_dtypes() -> None:
    """PLAYER_MATCH.coerce selects and pins the canonical dtypes."""
    frame = _player_match_frame(
        [
            {
                "season": "2024-25",
                "gw": 1,
                "element": 5,
                "opponent": 12,
                "is_home": True,
                "minutes": 90,
                "total_points": 6,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": datetime(2024, 8, 10, 15, 0),
                "extra": "ignored",
            }
        ]
    )
    result = PLAYER_MATCH.coerce(frame)
    assert result.columns == PLAYER_MATCH.columns
    assert result["gw"].dtype == pl.Int64
    assert result["opponent"].dtype == pl.Int64
    assert result["is_home"].dtype == pl.Boolean


def test_player_match_pk_disambiguates_double_gameweek(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Two fixtures for one (season, gw, element) coexist via distinct opponent."""
    connection = db
    frame = _player_match_frame(
        [
            {
                "season": "2024-25",
                "gw": 1,
                "element": 5,
                "opponent": 12,
                "is_home": True,
                "minutes": 90,
                "total_points": 6,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": datetime(2024, 8, 10, 15, 0),
            },
            {
                "season": "2024-25",
                "gw": 1,
                "element": 5,
                "opponent": 7,
                "is_home": False,
                "minutes": 70,
                "total_points": 2,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": datetime(2024, 8, 14, 19, 0),
            },
        ]
    )
    PLAYER_MATCH.upsert_current(connection, frame, "2024-25")
    row = connection.execute("SELECT COUNT(*) FROM player_match").fetchone()
    assert row is not None
    count = row[0]
    assert count == 2


def test_write_immutable_player_match_is_noop_when_present(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A second immutable write for an existing season is discarded."""
    connection = db
    frame = _player_match_frame(
        [
            {
                "season": "2023-24",
                "gw": 1,
                "element": 1,
                "opponent": 2,
                "is_home": True,
                "minutes": 90,
                "total_points": 3,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": datetime(2023, 8, 11, 19, 0),
            }
        ]
    )
    PLAYER_MATCH.write_immutable(connection, frame, "2023-24")
    assert PLAYER_MATCH.seasons_present(connection) == {"2023-24"}
    PLAYER_MATCH.write_immutable(connection, frame, "2023-24")
    row = connection.execute("SELECT COUNT(*) FROM player_match").fetchone()
    assert row is not None
    count = row[0]
    assert count == 1


def test_load_player_match_round_trips_ordered(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """PLAYER_MATCH.load returns every row ordered by the key."""
    connection = db
    frame = _player_match_frame(
        [
            {
                "season": "2024-25",
                "gw": 2,
                "element": 9,
                "opponent": 3,
                "is_home": True,
                "minutes": 45,
                "total_points": 1,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": datetime(2024, 8, 17, 15, 0),
            },
            {
                "season": "2024-25",
                "gw": 1,
                "element": 9,
                "opponent": 4,
                "is_home": False,
                "minutes": 90,
                "total_points": 5,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": datetime(2024, 8, 10, 15, 0),
            },
        ]
    )
    PLAYER_MATCH.upsert_current(connection, frame, "2024-25")
    out = PLAYER_MATCH.load(connection)
    assert out.columns == PLAYER_MATCH.columns
    assert out["gw"].to_list() == [1, 2]


def _availability_frame(season: str, chance: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [season],
            "gw": [1],
            "element": [10],
            "chance_of_playing_this_round": [chance],
        }
    )


def test_write_immutable_player_availability_inserts_once(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A completed season is inserted once and not duplicated on re-write."""
    connection = db
    PLAYER_AVAILABILITY.write_immutable(
        connection, _availability_frame("2022-23", 75), "2022-23"
    )
    # Second write with a different value must be ignored (already present).
    PLAYER_AVAILABILITY.write_immutable(
        connection, _availability_frame("2022-23", 0), "2022-23"
    )
    out = PLAYER_AVAILABILITY.load(connection)

    assert out.height == 1
    assert out["chance_of_playing_this_round"].to_list() == [75]


def test_upsert_current_player_availability_replaces_season(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Upsert replaces all rows for the season with the fresh frame."""
    connection = db
    PLAYER_AVAILABILITY.upsert_current(
        connection, _availability_frame("2025-26", 50), "2025-26"
    )
    PLAYER_AVAILABILITY.upsert_current(
        connection, _availability_frame("2025-26", 100), "2025-26"
    )
    out = PLAYER_AVAILABILITY.load(connection)

    assert out.height == 1
    assert out["chance_of_playing_this_round"].to_list() == [100]


from fantasy_football.storage.tables import (  # noqa: E402
    minutes_prediction_versions,
)


def _minutes_prediction_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows)


def test_get_connection_creates_minutes_prediction_table(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """get_connection creates an empty minutes_prediction table."""
    connection = db
    columns = [
        row[0]
        for row in connection.execute("DESCRIBE minutes_prediction").fetchall()
    ]
    assert columns == MINUTES_PREDICTION.columns


def test_coerce_minutes_prediction_selects_and_pins_dtypes() -> None:
    """Coerce drops extra columns and pins the canonical dtypes."""
    frame = _minutes_prediction_frame(
        [
            {
                "season": "2024-25",
                "gw": 1,
                "element": 5,
                "opponent": 12,
                "p_zero": 0.1,
                "p_partial": 0.2,
                "p_sixty_plus": 0.7,
                "expected_minutes": 58.5,
                "model_version": "3",
                "prediction_kind": "backfill",
                "snapshot_captured_at": None,
                "extra": "ignored",
            }
        ]
    )
    result = MINUTES_PREDICTION.coerce(frame)
    assert result.columns == MINUTES_PREDICTION.columns
    assert result["gw"].dtype == pl.Int64
    assert result["opponent"].dtype == pl.Int64
    assert result["p_sixty_plus"].dtype == pl.Float64
    assert result["model_version"].dtype == pl.Utf8


def test_minutes_prediction_pk_disambiguates_double_gameweek(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Two fixtures for one (season, gw, element) coexist via distinct opponent."""
    connection = db
    frame = _minutes_prediction_frame(
        [
            {
                "season": "2024-25",
                "gw": 1,
                "element": 5,
                "opponent": 12,
                "p_zero": 0.1,
                "p_partial": 0.2,
                "p_sixty_plus": 0.7,
                "expected_minutes": 58.5,
                "model_version": "3",
                "prediction_kind": "backfill",
                "snapshot_captured_at": None,
            },
            {
                "season": "2024-25",
                "gw": 1,
                "element": 5,
                "opponent": 7,
                "p_zero": 0.3,
                "p_partial": 0.3,
                "p_sixty_plus": 0.4,
                "expected_minutes": 39.0,
                "model_version": "3",
                "prediction_kind": "backfill",
                "snapshot_captured_at": None,
            },
        ]
    )
    MINUTES_PREDICTION.upsert_current(connection, frame, "2024-25")
    out = MINUTES_PREDICTION.load(connection)
    assert out.height == 2


def test_upsert_minutes_prediction_replaces_season(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Upsert replaces all rows for the season with the fresh frame."""
    connection = db

    def one_row(version: str, exp: float) -> pl.DataFrame:
        return _minutes_prediction_frame(
            [
                {
                    "season": "2025-26",
                    "gw": 1,
                    "element": 5,
                    "opponent": 12,
                    "p_zero": 0.1,
                    "p_partial": 0.2,
                    "p_sixty_plus": 0.7,
                    "expected_minutes": exp,
                    "model_version": version,
                    "prediction_kind": "backfill",
                    "snapshot_captured_at": None,
                }
            ]
        )

    MINUTES_PREDICTION.upsert_current(
        connection, one_row("1", 58.5), "2025-26"
    )
    MINUTES_PREDICTION.upsert_current(
        connection, one_row("2", 60.0), "2025-26"
    )
    out = MINUTES_PREDICTION.load(connection)
    assert out.height == 1
    assert out["model_version"].to_list() == ["2"]
    assert out["expected_minutes"].to_list() == [60.0]


def test_minutes_prediction_versions_returns_distinct(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Distinct model versions are returned, optionally filtered by season."""
    connection = db

    def row(season: str, version: str) -> pl.DataFrame:
        return _minutes_prediction_frame(
            [
                {
                    "season": season,
                    "gw": 1,
                    "element": 5,
                    "opponent": 12,
                    "p_zero": 0.1,
                    "p_partial": 0.2,
                    "p_sixty_plus": 0.7,
                    "expected_minutes": 58.5,
                    "model_version": version,
                    "prediction_kind": "backfill",
                    "snapshot_captured_at": None,
                }
            ]
        )

    MINUTES_PREDICTION.upsert_current(
        connection, row("2024-25", "1"), "2024-25"
    )
    MINUTES_PREDICTION.upsert_current(
        connection, row("2025-26", "2"), "2025-26"
    )
    all_versions = minutes_prediction_versions(connection)
    historic = minutes_prediction_versions(connection, seasons=["2024-25"])
    assert all_versions == {"1", "2"}
    assert historic == {"1"}


def test_reset_database_drops_minutes_prediction_rows(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """reset_database empties the minutes_prediction table."""
    connection = db
    MINUTES_PREDICTION.upsert_current(
        connection,
        _minutes_prediction_frame(
            [
                {
                    "season": "2025-26",
                    "gw": 1,
                    "element": 5,
                    "opponent": 12,
                    "p_zero": 0.1,
                    "p_partial": 0.2,
                    "p_sixty_plus": 0.7,
                    "expected_minutes": 58.5,
                    "model_version": "1",
                    "prediction_kind": "backfill",
                    "snapshot_captured_at": None,
                }
            ]
        ),
        "2025-26",
    )
    reset_database(connection)
    out = MINUTES_PREDICTION.load(connection)
    assert out.height == 0


def test_get_connection_creates_player_season_table(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """get_connection creates an empty player_season table with canonical columns."""
    connection = db
    columns = [
        row[0]
        for row in connection.execute("DESCRIBE player_season").fetchall()
    ]

    assert columns == [
        "season",
        "element",
        "player_code",
        "web_name",
        "first_name",
        "second_name",
        "position",
        "team_code",
        "birth_date",
        "region",
        "team_join_date",
    ]


def _player_season_frame(
    season: str,
    element: int,
    player_code: int,
    birth_date: date | None = None,
    team_join_date: date | None = None,
    web_name: str = "Salah",
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [season],
            "element": [element],
            "player_code": [player_code],
            "web_name": [web_name],
            "first_name": ["Mohamed"],
            "second_name": ["Salah"],
            "position": ["MID"],
            "team_code": [14],
            "birth_date": [birth_date],
            "region": [None],
            "team_join_date": [team_join_date],
        },
        schema_overrides={
            "birth_date": pl.Date,
            "team_join_date": pl.Date,
            "region": pl.Int64,
        },
    )


def test_write_immutable_player_season_skips_present_season(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A season already stored is not written a second time."""
    connection = db
    PLAYER_SEASON.write_immutable(
        connection, _player_season_frame("2023-24", 1, 111), "2023-24"
    )
    PLAYER_SEASON.write_immutable(
        connection,
        _player_season_frame("2023-24", 2, 222),
        "2023-24",
    )
    out = PLAYER_SEASON.load(connection)

    assert out.height == 1
    assert out["element"].to_list() == [1]


def test_upsert_current_player_season_replaces_season(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Upserting a season replaces its rows rather than appending."""
    connection = db
    PLAYER_SEASON.upsert_current(
        connection, _player_season_frame("2026-27", 1, 111), "2026-27"
    )
    PLAYER_SEASON.upsert_current(
        connection, _player_season_frame("2026-27", 9, 111), "2026-27"
    )
    out = PLAYER_SEASON.load(connection)

    assert out.height == 1
    assert out["element"].to_list() == [9]


def test_load_player_season_propagates_birth_date_across_seasons(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """birth_date observed in one season fills the same code's other seasons."""
    connection = db
    PLAYER_SEASON.write_immutable(
        connection,
        _player_season_frame("2021-22", 1, 111, birth_date=None),
        "2021-22",
    )
    PLAYER_SEASON.write_immutable(
        connection,
        _player_season_frame("2023-24", 5, 111, birth_date=date(1992, 6, 15)),
        "2023-24",
    )
    out = PLAYER_SEASON.load(connection).sort("season")

    assert out["birth_date"].to_list() == [
        date(1992, 6, 15),
        date(1992, 6, 15),
    ]


def test_load_player_season_does_not_propagate_season_varying_columns(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """team_join_date and web_name stay per-season; a null stays null."""
    connection = db
    PLAYER_SEASON.write_immutable(
        connection,
        _player_season_frame(
            "2021-22", 1, 111, team_join_date=None, web_name="M.Salah"
        ),
        "2021-22",
    )
    PLAYER_SEASON.write_immutable(
        connection,
        _player_season_frame(
            "2023-24",
            5,
            111,
            team_join_date=date(2017, 7, 1),
            web_name="Salah",
        ),
        "2023-24",
    )
    out = PLAYER_SEASON.load(connection).sort("season")

    assert out["team_join_date"].to_list() == [None, date(2017, 7, 1)]
    assert out["web_name"].to_list() == ["M.Salah", "Salah"]


def test_reset_database_drops_player_season_rows(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """reset_database empties the player_season table."""
    connection = db
    PLAYER_SEASON.write_immutable(
        connection, _player_season_frame("2023-24", 1, 111), "2023-24"
    )
    reset_database(connection)
    out = PLAYER_SEASON.load(connection)

    assert out.height == 0
