import logging
from pathlib import Path

import polars as pl
import pytest

from fantasy_football.features import transformation as data_transformation
from fantasy_football.features.transformation import (
    KNOWN_POSITIONS,
    add_rolling_identity_column,
    create_rolling_average_column,
    create_rolling_points_data,
    fill_missing_values_by_position,
    load_gw_data,
    rolling_column_name,
)
from fantasy_football.storage import database
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import PLAYER_SEASON, PLAYER_WEEK


@pytest.fixture
def sample_gw_data() -> pl.DataFrame:
    """Create a sample gameweek data DataFrame for testing.

    Returns
    -------
    pl.DataFrame
        A DataFrame containing sample gameweek data.
    """
    return pl.DataFrame(
        {
            "season": ["2020-21", "2020-21", "2020-21", "2021-22", "2021-22"],
            "name": ["Player1", "Player1", "Player1", "Player2", "Player2"],
            "position": ["GK", "GK", "GK", "DEF", "DEF"],
            "bonus": [1, 2, 0, 1, 3],
            "element": [1, 1, 1, 2, 2],
            "minutes": [90, 90, 90, 90, 90],
            "round": [1, 2, 3, 1, 2],
            "total_points": [6, 8, 4, 7, 9],
            "gw": [1, 2, 3, 1, 2],
        }
    )


def _seed_player_week_db(db_path: Path) -> None:
    """Seed a temp DB with one historic (GKP) season and one current season."""
    historic = pl.DataFrame(
        {
            "season": ["2020-21", "2020-21"],
            "gw": [1, 2],
            "element": [1, 1],
            "name": ["Player1", "Player1"],
            "position": ["GKP", "GKP"],
            "team": ["Arsenal", "Arsenal"],
            "bonus": [1, 2],
            "minutes": [90, 90],
            "round": [1, 2],
            "total_points": [6, 8],
            "value": [50, 50],
        }
    )
    current = pl.DataFrame(
        {
            "season": ["2025-26", "2025-26"],
            "gw": [1, 2],
            "element": [2, 2],
            "name": ["Player2", "Player2"],
            "position": ["GK", "GK"],
            "team": ["Chelsea", "Chelsea"],
            "bonus": [1, 2],
            "minutes": [90, 90],
            "round": [1, 2],
            "total_points": [7, 9],
            "value": [45, 45],
        }
    )
    connection = get_connection(db_path)
    try:
        PLAYER_WEEK.write_immutable(connection, historic, "2020-21")
        PLAYER_WEEK.upsert_current(connection, current, "2025-26")
    finally:
        connection.close()


def test_load_gw_data_reads_all_seasons_from_db(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """load_gw_data returns every stored season with GKP collapsed to GK."""
    db_path = tmp_path / "t.duckdb"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    _seed_player_week_db(db_path)

    result = load_gw_data()

    assert isinstance(result, pl.DataFrame)
    assert set(result["season"].unique().to_list()) == {"2020-21", "2025-26"}
    assert result.filter(pl.col("position") == "GKP").height == 0
    assert result.filter(pl.col("position") == "GK").height == 4
    assert result.filter(pl.col("name") == "Player1")[
        "team"
    ].unique().to_list() == ["Arsenal"]


def test_create_rolling_average_column(sample_gw_data: pl.DataFrame) -> None:
    """Test the create_rolling_average_column function.

    Parameters
    ----------
    sample_gw_data : pl.DataFrame
        Sample gameweek data for testing.
    """
    result = create_rolling_average_column(
        sample_gw_data,
        grouping_columns=["name"],
        rolling_column="total_points",
        rolling_window=2,
    )

    assert isinstance(result, pl.DataFrame)
    assert rolling_column_name("total_points", 2) in result.columns

    # min_periods=1 means rows with fewer than `rolling_window` prior games
    # now get a mean over whatever exists, rather than null. The first row
    # per player is therefore its own value rather than None.
    player1_values = (
        result.filter(pl.col("name") == "Player1")
        .sort(["season", "gw"])[rolling_column_name("total_points", 2)]
        .to_list()
    )
    assert player1_values == [6.0, 7.0, 6.0]

    player2_values = (
        result.filter(pl.col("name") == "Player2")
        .sort(["season", "gw"])[rolling_column_name("total_points", 2)]
        .to_list()
    )
    assert player2_values == [7.0, 8.0]


def test_create_rolling_average_column_partitions_by_group() -> None:
    """Verify rolling averages do not leak across the grouping column.

    Uses an interleaved fixture (Player1 at odd gameweeks, Player2 at even)
    so that if ``.over("name")`` were removed, global rolling would pick up
    cross-player values and produce different numbers than per-player rolling.
    """
    interleaved = pl.DataFrame(
        {
            "season": ["2020-21"] * 6,
            "name": [
                "Player1",
                "Player2",
                "Player1",
                "Player2",
                "Player1",
                "Player2",
            ],
            "position": ["GK"] * 6,
            "gw": [1, 2, 3, 4, 5, 6],
            "total_points": [10, 100, 20, 200, 30, 300],
        }
    )

    result = create_rolling_average_column(
        interleaved,
        grouping_columns=["name"],
        rolling_column="total_points",
        rolling_window=2,
    )

    player1_values = (
        result.filter(pl.col("name") == "Player1")
        .sort("gw")[rolling_column_name("total_points", 2)]
        .to_list()
    )
    player2_values = (
        result.filter(pl.col("name") == "Player2")
        .sort("gw")[rolling_column_name("total_points", 2)]
        .to_list()
    )
    # Per-player means: P1 = [10, 15, 25], P2 = [100, 150, 250]
    # (min_periods=1 means the first row is its own value, not null).
    # Without partitioning, the first rolling value would mix 10 and 100
    # (= 55.0), which is what this test catches.
    assert player1_values == [10.0, 15.0, 25.0]
    assert player2_values == [100.0, 150.0, 250.0]


def test_fill_missing_values_by_position(sample_gw_data: pl.DataFrame) -> None:
    """Test the fill_missing_values_by_position function.

    Parameters
    ----------
    sample_gw_data : pl.DataFrame
        Sample gameweek data for testing.
    """
    # Null only Player1's gw=1 row so a GK average is still computable
    data_with_nulls = sample_gw_data.with_columns(
        pl.when((pl.col("name") == "Player1") & (pl.col("gw") == 1))
        .then(None)
        .otherwise(pl.col("total_points"))
        .alias("total_points")
    )

    result = fill_missing_values_by_position(data_with_nulls, "total_points")

    # Assertions
    assert isinstance(result, pl.DataFrame)
    assert not result["total_points"].is_null().any()
    # GK average of non-null values (8 + 4) / 2 = 6.0, used to fill gw=1
    assert (
        result.filter((pl.col("name") == "Player1") & (pl.col("gw") == 1))
        .select("total_points")
        .item(0, 0)
        == 6.0
    )


def test_fill_missing_values_by_position_with_absent_position() -> None:
    """A hard-coded position with zero rows in the input should be a no-op.

    The function iterates over ``["GK", "DEF", "MID", "FWD"]``. If the data
    contains only some of those, the iterations for the absent positions must
    not raise and must not modify any rows.
    """
    data = pl.DataFrame(
        {
            "name": ["P1", "P2"],
            "position": ["GK", "GK"],
            "total_points": [None, 10],
        }
    )

    result = fill_missing_values_by_position(data, "total_points")

    assert result["total_points"].to_list() == [10.0, 10.0]


def test_fill_missing_values_by_position_all_null_for_position() -> None:
    """Document current behavior when every value in a position is null.

    ``position_data.select(...).mean().item(0, 0)`` returns ``None`` when all
    inputs are null, and the function then "fills" nulls with null — leaving
    them unchanged. This test pins that behavior so a future fix is a
    deliberate decision rather than an accidental change.
    """
    data = pl.DataFrame(
        {
            "name": ["P1", "P2", "P3"],
            "position": ["GK", "GK", "DEF"],
            "total_points": [None, None, 5],
        }
    )

    result = fill_missing_values_by_position(data, "total_points")

    gk_values = result.filter(pl.col("position") == "GK")[
        "total_points"
    ].to_list()
    assert gk_values == [None, None]
    assert result.filter(pl.col("position") == "DEF")[
        "total_points"
    ].to_list() == [5]


def test_fill_missing_values_by_position_warns_on_unknown_position(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown positions are not filled, but a warning is logged.

    The function only iterates over ``["GK", "DEF", "MID", "FWD"]``. Any other
    position string (e.g. ``"MNG"`` for the manager chip introduced in
    2024-25) leaves its nulls in place, but the function must emit a logged
    warning naming the unknown position(s) so silent gaps in the output are
    visible.
    """
    unknown = "MNG"
    assert unknown not in KNOWN_POSITIONS  # sanity-check the fixture's premise

    data = pl.DataFrame(
        {
            "name": ["P1", "P2"],
            "position": [unknown, unknown],
            "total_points": [None, 12],
        }
    )

    with caplog.at_level(
        logging.WARNING, logger="fantasy_football.features.transformation"
    ):
        result = fill_missing_values_by_position(data, "total_points")

    assert result["total_points"].to_list() == [None, 12]
    assert any(
        record.levelno == logging.WARNING and unknown in record.getMessage()
        for record in caplog.records
    )


def test_fill_missing_values_by_position_no_warning_for_known_positions(
    sample_gw_data: pl.DataFrame,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Inputs that only contain known positions must not trigger a warning."""
    with caplog.at_level(
        logging.WARNING, logger="fantasy_football.features.transformation"
    ):
        fill_missing_values_by_position(sample_gw_data, "total_points")

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


def _player_season_frame(
    season: str, element: int, player_code: int
) -> pl.DataFrame:
    """Build a minimal ``player_season`` row so ``player_code`` resolves.

    Parameters
    ----------
    season : str
        The season string, e.g. "2025-26".
    element : int
        The FPL element ID for this row.
    player_code : int
        The cross-season identity to attach to ``element`` for ``season``.

    Returns
    -------
    pl.DataFrame
        A single-row ``player_season``-shaped frame.
    """
    return pl.DataFrame(
        {
            "season": [season],
            "element": [element],
            "player_code": [player_code],
            "web_name": ["Salah"],
            "first_name": ["Mohamed"],
            "second_name": ["Salah"],
            "position": ["MID"],
            "team_code": [14],
            "birth_date": [None],
            "region": [None],
            "team_join_date": [None],
        },
        schema_overrides={
            "birth_date": pl.Date,
            "team_join_date": pl.Date,
            "region": pl.Int64,
        },
    )


def _seed_and_point_at(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gw_data: pl.DataFrame,
    player_code: int = 111,
) -> None:
    """Seed a temporary DuckDB with player-week data and redirect the pipeline at it.

    Writes ``gw_data`` into ``player_week`` season by season, and adds a
    matching ``player_season`` row for every ``(season, element)`` pair
    present -- all sharing ``player_code`` -- so ``player_code`` resolves for
    every row. ``database.DATABASE_PATH`` is monkeypatched to a tmp DuckDB
    file and ``transformation.TRANSFORMED_DATA_FOLDER`` to a tmp directory,
    so the pipeline reads and writes in isolation from the real database.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory path provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to redirect the DB and output folder.
    gw_data : pl.DataFrame
        Player-week rows to seed, in ``player_week`` shape, spanning one or
        more seasons.
    player_code : int, optional
        The cross-season identity given to every seeded player_season row.
        Defaults to 111. Callers seeding more than one distinct player should
        not rely on the default.
    """
    db_path = tmp_path / "t.duckdb"
    transformed_data = tmp_path / "transformed"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    monkeypatch.setattr(
        data_transformation, "TRANSFORMED_DATA_FOLDER", transformed_data
    )

    connection = get_connection(db_path)
    try:
        for season in gw_data["season"].unique(maintain_order=True).to_list():
            season_data = gw_data.filter(pl.col("season") == season)
            PLAYER_WEEK.write_immutable(connection, season_data, season)
            for element in (
                season_data["element"].unique(maintain_order=True).to_list()
            ):
                PLAYER_SEASON.write_immutable(
                    connection,
                    _player_season_frame(season, element, player_code),
                    season,
                )
    finally:
        connection.close()


@pytest.mark.parametrize("rolling_window", [2, 3, 5])
def test_create_rolling_points_data_respects_rolling_window(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    rolling_window: int,
) -> None:
    """Verify the output column name and values track the rolling_window arg.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory path provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to redirect the DB and output folder.
    rolling_window : int
        The rolling window size to test.
    """
    db_path = tmp_path / "t.duckdb"
    transformed_data = tmp_path / "transformed"
    current_season = "2025-26"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    monkeypatch.setattr(
        data_transformation, "TRANSFORMED_DATA_FOLDER", transformed_data
    )
    assert not transformed_data.exists()

    historic = pl.DataFrame(
        {
            "season": ["2020-21"] * 6,
            "gw": [1, 2, 3, 4, 5, 6],
            "element": [1] * 6,
            "name": ["Player1"] * 6,
            "position": ["GK"] * 6,
            "team": ["Arsenal"] * 6,
            "bonus": [0, 1, 2, 0, 1, 2],
            "minutes": [90] * 6,
            "round": [1, 2, 3, 4, 5, 6],
            "total_points": [2, 4, 6, 8, 10, 12],
            "value": [50] * 6,
        }
    )
    connection = get_connection(db_path)
    try:
        PLAYER_WEEK.write_immutable(connection, historic, "2020-21")
    finally:
        connection.close()

    create_rolling_points_data(current_season, rolling_window=rolling_window)

    result = pl.read_csv(transformed_data / "rolling_points.csv")
    expected_column = rolling_column_name("total_points", rolling_window)
    assert expected_column in result.columns

    player1_tail_points = (
        result.filter(pl.col("name") == "Player1")
        .sort(["season", "gw"])
        .tail(rolling_window)["total_points"]
        .to_list()
    )
    expected_mean = sum(player1_tail_points) / rolling_window
    actual = result.filter(pl.col("name") == "Player1").sort(["season", "gw"])[
        expected_column
    ][-1]
    assert actual == pytest.approx(expected_mean)


def test_rolling_average_separates_players_sharing_a_name() -> None:
    """Two distinct player_codes with the same name keep separate series.

    This is the regression test for the original bug: the window was keyed on
    the display name, so namesakes were pooled into one rolling series.
    """
    data = pl.DataFrame(
        {
            "season": ["2023-24"] * 4,
            "gw": [1, 2, 1, 2],
            "name": ["Danny Ward"] * 4,
            "player_code": [111, 111, 222, 222],
            "total_points": [10, 10, 2, 2],
        }
    )
    out = create_rolling_average_column(
        data, ["player_code", "season"], "total_points", 2
    )
    column = rolling_column_name("total_points", 2)
    values = out.filter(pl.col("player_code") == 222)[column].to_list()

    assert values == [2.0, 2.0]


def test_rolling_average_restarts_when_season_is_in_the_grouping() -> None:
    """Including ``season`` in the partition restarts the window each season.

    This covers the grouping argument itself. The points pipeline no longer
    passes ``season`` -- see
    ``test_create_rolling_points_data_spans_the_season_boundary``.
    """
    data = pl.DataFrame(
        {
            "season": ["2023-24", "2023-24", "2024-25"],
            "gw": [1, 2, 1],
            "name": ["Salah"] * 3,
            "player_code": [111, 111, 111],
            "total_points": [10, 10, 2],
        }
    )
    out = create_rolling_average_column(
        data, ["player_code", "season"], "total_points", 2
    )
    column = rolling_column_name("total_points", 2)

    assert out.sort(["season", "gw"])[column].to_list() == [10.0, 10.0, 2.0]


def test_create_rolling_points_data_spans_the_season_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A new season's first gameweek averages in the prior season's tail.

    One player scores 10 in each of the last two gameweeks of 2024-25, then
    2 in 2025-26 GW1. With a window of 2, that GW1 row must be mean(10, 2)
    = 6.0 -- the window reaching back across the summer -- rather than 2.0.
    """
    _seed_and_point_at(
        tmp_path,
        monkeypatch,
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25", "2025-26"],
                "gw": [37, 38, 1],
                "element": [1, 1, 1],
                "name": ["Salah"] * 3,
                "position": ["MID"] * 3,
                "team": ["Liverpool"] * 3,
                "bonus": [0] * 3,
                "minutes": [90] * 3,
                "round": [37, 38, 1],
                "total_points": [10, 10, 2],
                "value": [130] * 3,
            }
        ),
    )

    create_rolling_points_data("2025-26", rolling_window=2)

    out = pl.read_csv(
        data_transformation.TRANSFORMED_DATA_FOLDER / "rolling_points.csv"
    )
    column = rolling_column_name("total_points", 2)
    gw1 = out.filter((pl.col("season") == "2025-26") & (pl.col("gw") == 1))[
        column
    ].item()

    assert gw1 == 6.0


def test_add_rolling_identity_column_gives_distinct_fallbacks_for_null_codes() -> (
    None
):
    """Two null-player_code rows for different elements get different keys.

    ``.over()`` pools every null value into one partition, so a fallback is
    required for rows with no ``player_code``. This checks the fallback is
    keyed on ``element`` (distinct rows stay distinct) and cannot collide
    with a real ``player_code`` string.
    """
    data = pl.DataFrame(
        {
            "player_code": [None, None, 111],
            "element": [1, 2, 111],
        }
    )

    result = add_rolling_identity_column(data)

    identities = result["rolling_identity"].to_list()
    assert len(set(identities)) == 3
    # The real player_code's identity is its bare digit string...
    assert identities[2] == "111"
    # ...which cannot equal either fallback, even though element == 111 here.
    assert identities[0] != identities[2]
    assert identities[1] != identities[2]


def test_create_rolling_points_data_separates_players_with_null_player_code(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Rows with no player_season identity row must not be pooled together.

    ``player_code`` is null whenever a season has no matching
    ``player_season`` row -- which happens by design when
    ``load_player_identity_data`` catches a failed source fetch and skips
    that season. Grouping the rolling window directly on ``player_code``
    would pool every null-coded player in that season into one shared
    series. This seeds two distinct players in the same season with no
    ``player_season`` rows at all (so both get a null ``player_code``) and
    clearly different points, and asserts their rolling series stay
    separate rather than being averaged together.
    """
    db_path = tmp_path / "t.duckdb"
    transformed_data = tmp_path / "transformed"
    current_season = "2025-26"
    monkeypatch.setattr(database, "DATABASE_PATH", db_path)
    monkeypatch.setattr(
        data_transformation, "TRANSFORMED_DATA_FOLDER", transformed_data
    )

    historic = pl.DataFrame(
        {
            "season": ["2020-21"] * 4,
            "gw": [1, 2, 1, 2],
            "element": [1, 1, 2, 2],
            "name": ["Player1", "Player1", "Player2", "Player2"],
            "position": ["GK", "GK", "GK", "GK"],
            "team": ["Arsenal", "Arsenal", "Chelsea", "Chelsea"],
            "bonus": [0, 0, 0, 0],
            "minutes": [90, 90, 90, 90],
            "round": [1, 2, 1, 2],
            "total_points": [10, 10, 2, 2],
            "value": [50, 50, 50, 50],
        }
    )
    connection = get_connection(db_path)
    try:
        PLAYER_WEEK.write_immutable(connection, historic, "2020-21")
    finally:
        connection.close()

    create_rolling_points_data(current_season, rolling_window=2)

    result = pl.read_csv(transformed_data / "rolling_points.csv")
    column = rolling_column_name("total_points", 2)
    player1_values = (
        result.filter(pl.col("element") == 1).sort("gw")[column].to_list()
    )
    player2_values = (
        result.filter(pl.col("element") == 2).sort("gw")[column].to_list()
    )

    assert player1_values == [10.0, 10.0]
    assert player2_values == [2.0, 2.0]
