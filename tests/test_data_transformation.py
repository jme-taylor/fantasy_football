from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
import pytest

from fantasy_football.data_transformation import (
    create_rolling_average_column,
    create_rolling_points_data,
    fill_missing_values_by_position,
    load_gw_data,
)

if TYPE_CHECKING:
    from fantasy_football.constants import DATA_FOLDER

@pytest.fixture
def sample_gw_data() -> pl.DataFrame:
    """Create a sample gameweek data DataFrame for testing.
    
    Returns
    -------
    pl.DataFrame
        A DataFrame containing sample gameweek data.
    """
    return pl.DataFrame({
        "season": ["2020-21", "2020-21", "2020-21", "2021-22", "2021-22"],
        "name": ["Player1", "Player1", "Player1", "Player2", "Player2"],
        "position": ["GKP", "GKP", "GKP", "DEF", "DEF"],
        "bonus": [1, 2, 0, 1, 3],
        "element": [1, 1, 1, 2, 2],
        "minutes": [90, 90, 90, 90, 90],
        "round": [1, 2, 3, 1, 2],
        "total_points": [6, 8, 4, 7, 9],
        "gw": [1, 2, 3, 1, 2],
    })

def test_load_gw_data(tmp_path: Path) -> None:
    """Test the load_gw_data function.
    
    Parameters
    ----------
    tmp_path : Path
        A temporary directory path provided by pytest.
    """
    # Create temporary data structure
    raw_data = tmp_path / "raw"
    current_season = "2023-24"
    season_data = raw_data / current_season / "gws"
    season_data.mkdir(parents=True)
    
    # Create sample data files
    previous_seasons_data = pl.DataFrame({
        "season_x": ["2020-21", "2020-21"],
        "name": ["Player1", "Player1"],
        "position": ["GKP", "GKP"],
        "bonus": [1, 2],
        "element": [1, 1],
        "minutes": [90, 90],
        "round": [1, 2],
        "total_points": [6, 8],
        "GW": [1, 2],
    })
    previous_seasons_data.write_csv(raw_data / "cleaned_merged_seasons.csv")
    
    current_season_data = pl.DataFrame({
        "name": ["Player2", "Player2"],
        "position": ["GKP", "GKP"],
        "bonus": [1, 2],
        "element": [2, 2],
        "minutes": [90, 90],
        "round": [1, 2],
        "total_points": [7, 9],
        "GW": [1, 2],
    })
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
    assert result.filter(pl.col("position") == "GKP").height == 0  # Should be converted to "GK"
    assert result.filter(pl.col("position") == "GK").height > 0

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
        rolling_window=2
    )
    
    # Assertions
    assert isinstance(result, pl.DataFrame)
    assert "total_points_rolling_2" in result.columns
    assert result.filter(pl.col("name") == "Player1").select("total_points_rolling_2").mean().item(0, 0) == 7.0  # (8 + 4) / 2

def test_fill_missing_values_by_position(sample_gw_data: pl.DataFrame) -> None:
    """Test the fill_missing_values_by_position function.
    
    Parameters
    ----------
    sample_gw_data : pl.DataFrame
        Sample gameweek data for testing.
    """
    # Add some missing values
    data_with_nulls = sample_gw_data.with_columns(
        pl.when(pl.col("name") == "Player1").then(None).otherwise(pl.col("total_points")).alias("total_points")
    )
    
    result = fill_missing_values_by_position(data_with_nulls, "total_points")
    
    # Assertions
    assert isinstance(result, pl.DataFrame)
    assert not result["total_points"].is_null().any()
    assert result.filter(pl.col("name") == "Player1").select("total_points").mean().item(0, 0) == 6.0  # Average of GK position

def test_create_rolling_points_data(tmp_path: Path) -> None:
    """Test the create_rolling_points_data function.
    
    Parameters
    ----------
    tmp_path : Path
        A temporary directory path provided by pytest.
    """
    # Create temporary data structure
    raw_data = tmp_path / "raw"
    transformed_data = tmp_path / "transformed"
    current_season = "2023-24"
    season_data = raw_data / current_season / "gws"
    season_data.mkdir(parents=True)
    transformed_data.mkdir(parents=True)
    
    # Create sample data files
    previous_seasons_data = pl.DataFrame({
        "season_x": ["2020-21", "2020-21"],
        "name": ["Player1", "Player1"],
        "position": ["GKP", "GKP"],
        "bonus": [1, 2],
        "element": [1, 1],
        "minutes": [90, 90],
        "round": [1, 2],
        "total_points": [6, 8],
        "GW": [1, 2],
    })
    previous_seasons_data.write_csv(raw_data / "cleaned_merged_seasons.csv")
    
    current_season_data = pl.DataFrame({
        "name": ["Player2", "Player2"],
        "position": ["GKP", "GKP"],
        "bonus": [1, 2],
        "element": [2, 2],
        "minutes": [90, 90],
        "round": [1, 2],
        "total_points": [7, 9],
        "GW": [1, 2],
    })
    current_season_data.write_csv(season_data / "merged_gw.csv")
    
    # Test the function
    create_rolling_points_data(current_season, rolling_window=5)
    
    # Assertions
    result_file = transformed_data / "rolling_points.csv"
    assert result_file.exists()
    result = pl.read_csv(result_file)
    assert "total_points_rolling_5" in result.columns 