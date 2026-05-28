import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from fantasy_football import data_transformation
from fantasy_football.data_transformation import (
    KNOWN_POSITIONS,
    create_rolling_average_column,
    create_rolling_points_data,
    fill_missing_values_by_position,
    load_gw_data,
    rolling_column_name,
)

if TYPE_CHECKING:
    pass


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


def test_load_gw_data(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test the load_gw_data function.

    Parameters
    ----------
    tmp_path : Path
        A temporary directory path provided by pytest.
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to redirect the module's data folders.
    """
    # Create temporary data structure
    raw_data = tmp_path / "raw"
    current_season = "2023-24"
    season_data = raw_data / current_season / "gws"
    season_data.mkdir(parents=True)
    monkeypatch.setattr(data_transformation, "RAW_DATA_FOLDER", raw_data)

    # Create sample data files
    previous_seasons_data = pl.DataFrame(
        {
            "season_x": ["2020-21", "2020-21"],
            "name": ["Player1", "Player1"],
            "position": ["GKP", "GKP"],
            "team_x": ["Arsenal", "Arsenal"],
            "bonus": [1, 2],
            "element": [1, 1],
            "minutes": [90, 90],
            "round": [1, 2],
            "total_points": [6, 8],
            "GW": [1, 2],
        }
    )
    previous_seasons_data.write_csv(raw_data / "cleaned_merged_seasons.csv")

    current_season_data = pl.DataFrame(
        {
            "name": ["Player2", "Player2"],
            "position": ["GK", "GK"],
            "team": ["Chelsea", "Chelsea"],
            "bonus": [1, 2],
            "element": [2, 2],
            "minutes": [90, 90],
            "round": [1, 2],
            "total_points": [7, 9],
            "GW": [1, 2],
        }
    )
    current_season_data.write_csv(season_data / "merged_gw.csv")

    # Test the function
    result = load_gw_data(current_season)

    # Assertions
    assert isinstance(result, pl.DataFrame)
    assert "season" in result.columns
    assert "name" in result.columns
    assert "position" in result.columns
    assert "total_points" in result.columns
    assert "gw" in result.columns
    # Row count: 2 previous + 2 current
    assert result.height == 4
    # GKP collapsed into GK
    assert result.filter(pl.col("position") == "GKP").height == 0
    assert result.filter(pl.col("position") == "GK").height == 4
    # Previous-season rows keep their original season, current-season rows
    # have ``current_season`` written into the season column.
    assert result.filter(pl.col("name") == "Player1")[
        "season"
    ].unique().to_list() == ["2020-21"]
    assert result.filter(pl.col("name") == "Player2")[
        "season"
    ].unique().to_list() == [current_season]
    assert "team" in result.columns
    assert result.filter(pl.col("name") == "Player1")[
        "team"
    ].unique().to_list() == ["Arsenal"]
    assert result.filter(pl.col("name") == "Player2")[
        "team"
    ].unique().to_list() == ["Chelsea"]


def test_load_gw_data_missing_previous_seasons_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing previous-seasons CSV must raise.

    Rather than silently dropping previous-season history, the function should
    surface a ``FileNotFoundError`` when ``cleaned_merged_seasons.csv`` is
    absent.
    """
    raw_data = tmp_path / "raw"
    current_season = "2023-24"
    season_data = raw_data / current_season / "gws"
    season_data.mkdir(parents=True)
    monkeypatch.setattr(data_transformation, "RAW_DATA_FOLDER", raw_data)

    pl.DataFrame(
        {
            "name": ["Player2"],
            "position": ["GK"],
            "bonus": [1],
            "element": [2],
            "minutes": [90],
            "round": [1],
            "total_points": [7],
            "GW": [1],
        }
    ).write_csv(season_data / "merged_gw.csv")

    with pytest.raises(FileNotFoundError):
        load_gw_data(current_season)


def test_load_gw_data_missing_current_season_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Missing current-season CSV must raise.

    Rather than returning only the historical data, the function should
    surface a ``FileNotFoundError`` when ``merged_gw.csv`` is absent.
    """
    raw_data = tmp_path / "raw"
    current_season = "2023-24"
    (raw_data / current_season / "gws").mkdir(parents=True)
    monkeypatch.setattr(data_transformation, "RAW_DATA_FOLDER", raw_data)

    pl.DataFrame(
        {
            "season_x": ["2020-21"],
            "name": ["Player1"],
            "position": ["GKP"],
            "team_x": ["Arsenal"],
            "bonus": [1],
            "element": [1],
            "minutes": [90],
            "round": [1],
            "total_points": [6],
            "GW": [1],
        }
    ).write_csv(raw_data / "cleaned_merged_seasons.csv")

    with pytest.raises(FileNotFoundError):
        load_gw_data(current_season)


def test_load_gw_data_schema_drift_in_previous_seasons(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Schema drift in the previous-seasons CSV must raise.

    If a required column (here ``total_points``) is missing, the function
    should surface a Polars ``ColumnNotFoundError`` rather than silently
    returning partial data.
    """
    raw_data = tmp_path / "raw"
    current_season = "2023-24"
    season_data = raw_data / current_season / "gws"
    season_data.mkdir(parents=True)
    monkeypatch.setattr(data_transformation, "RAW_DATA_FOLDER", raw_data)

    # ``total_points`` deliberately omitted from previous-seasons data.
    pl.DataFrame(
        {
            "season_x": ["2020-21"],
            "name": ["Player1"],
            "position": ["GKP"],
            "bonus": [1],
            "element": [1],
            "minutes": [90],
            "round": [1],
            "GW": [1],
        }
    ).write_csv(raw_data / "cleaned_merged_seasons.csv")

    pl.DataFrame(
        {
            "name": ["Player2"],
            "position": ["GK"],
            "bonus": [1],
            "element": [2],
            "minutes": [90],
            "round": [1],
            "total_points": [7],
            "GW": [1],
        }
    ).write_csv(season_data / "merged_gw.csv")

    with pytest.raises(pl.exceptions.ColumnNotFoundError):
        load_gw_data(current_season)


def test_create_rolling_average_column(sample_gw_data: pl.DataFrame) -> None:
    """Test the create_rolling_average_column function.

    Parameters
    ----------
    sample_gw_data : pl.DataFrame
        Sample gameweek data for testing.
    """
    result = create_rolling_average_column(
        sample_gw_data,
        grouping_column="name",
        rolling_column="total_points",
        rolling_window=2,
    )

    assert isinstance(result, pl.DataFrame)
    assert rolling_column_name("total_points", 2) in result.columns

    player1_values = (
        result.filter(pl.col("name") == "Player1")
        .sort(["season", "gw"])[rolling_column_name("total_points", 2)]
        .to_list()
    )
    assert player1_values == [None, 7.0, 6.0]

    player2_values = (
        result.filter(pl.col("name") == "Player2")
        .sort(["season", "gw"])[rolling_column_name("total_points", 2)]
        .to_list()
    )
    assert player2_values == [None, 8.0]


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
        grouping_column="name",
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
    # Per-player means: P1 = [None, 15, 25], P2 = [None, 150, 250]
    # Without partitioning, the first non-null rolling value would mix
    # 10 and 100 (= 55.0), which is what this test catches.
    assert player1_values == [None, 15.0, 25.0]
    assert player2_values == [None, 150.0, 250.0]


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
        logging.WARNING, logger="fantasy_football.data_transformation"
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
        logging.WARNING, logger="fantasy_football.data_transformation"
    ):
        fill_missing_values_by_position(sample_gw_data, "total_points")

    assert not [r for r in caplog.records if r.levelno == logging.WARNING]


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
        Pytest monkeypatch fixture used to redirect the module's data folders.
    rolling_window : int
        The rolling window size to test.
    """
    raw_data = tmp_path / "raw"
    transformed_data = tmp_path / "transformed"
    current_season = "2023-24"
    season_data = raw_data / current_season / "gws"
    season_data.mkdir(parents=True)
    monkeypatch.setattr(data_transformation, "RAW_DATA_FOLDER", raw_data)
    monkeypatch.setattr(
        data_transformation, "TRANSFORMED_DATA_FOLDER", transformed_data
    )
    assert not transformed_data.exists()

    previous_seasons_data = pl.DataFrame(
        {
            "season_x": ["2020-21"] * 6,
            "name": ["Player1"] * 6,
            "position": ["GK"] * 6,
            "team_x": ["Arsenal"] * 6,
            "bonus": [0, 1, 2, 0, 1, 2],
            "element": [1] * 6,
            "minutes": [90] * 6,
            "round": [1, 2, 3, 4, 5, 6],
            "total_points": [2, 4, 6, 8, 10, 12],
            "GW": [1, 2, 3, 4, 5, 6],
        }
    )
    previous_seasons_data.write_csv(raw_data / "cleaned_merged_seasons.csv")

    current_season_data = pl.DataFrame(
        {
            "name": ["Player2"] * 6,
            "position": ["GK"] * 6,
            "team": ["Chelsea"] * 6,
            "bonus": [0, 1, 2, 0, 1, 2],
            "element": [2] * 6,
            "minutes": [90] * 6,
            "round": [1, 2, 3, 4, 5, 6],
            "total_points": [1, 3, 5, 7, 9, 11],
            "GW": [1, 2, 3, 4, 5, 6],
        }
    )
    current_season_data.write_csv(season_data / "merged_gw.csv")

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
