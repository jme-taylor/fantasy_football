from typing import TYPE_CHECKING

import polars as pl
import pytest

from fantasy_football.fpl_types import FplPlayer, FplFixture, FplFixtures
from fantasy_football.prediction import get_player_fixtures, get_player_rolling_points

@pytest.fixture
def sample_fixtures() -> FplFixtures:
    """Create sample fixtures for testing.
    
    Returns
    -------
    FplFixtures
        A list of fixtures for testing.
    """
    return FplFixtures(
        fixtures=[
            # Home fixture for team 1 (finished)
            FplFixture(
                code=1001,
                event=1,
                finished=True,
                id=1,
                kickoff_time="2023-08-12T14:00:00Z",
                team_a=2,
                team_a_difficulty=3,
                team_h=1,
                team_h_difficulty=2,
            ),
            # Away fixture for team 1 (not finished)
            FplFixture(
                code=1002,
                event=2,
                finished=False,
                id=2,
                kickoff_time="2023-08-19T14:00:00Z",
                team_a=1,
                team_a_difficulty=3,
                team_h=3,
                team_h_difficulty=2,
            ),
            # Home fixture for team 1 (not finished)
            FplFixture(
                code=1003,
                event=3,
                finished=False,
                id=3,
                kickoff_time="2023-08-26T14:00:00Z",
                team_a=4,
                team_a_difficulty=2,
                team_h=1,
                team_h_difficulty=4,
            ),
            # Fixture not involving team 1 (not finished)
            FplFixture(
                code=1004,
                event=4,
                finished=False,
                id=4,
                kickoff_time="2023-09-02T14:00:00Z",
                team_a=3,
                team_a_difficulty=2,
                team_h=2,
                team_h_difficulty=3,
            ),
        ]
    )

@pytest.fixture
def sample_player() -> FplPlayer:
    """Create a sample player for testing.
    
    Returns
    -------
    FplPlayer
        A sample player for testing.
    """
    return FplPlayer(
        id=101,
        first_name="Test",
        second_name="Player",
        web_name="T. Player",
        selected_by_percent=10.5,
        now_cost=75,
        team_id=1,
        element_type=3
    )

@pytest.fixture
def sample_rolling_data() -> pl.DataFrame:
    """Create sample rolling data for testing.
    
    Returns
    -------
    pl.DataFrame
        Sample rolling points data.
    """
    return pl.DataFrame({
        "season": ["2024-25", "2024-25", "2023-24"],
        "element": [101, 101, 101],
        "gw": [2, 1, 38],
        "total_points_rolling_5": [6.5, 4.2, 5.8]
    })

def test_get_player_fixtures(sample_fixtures: FplFixtures, sample_player: FplPlayer) -> None:
    """Test the get_player_fixtures function.
    
    Parameters
    ----------
    sample_fixtures : FplFixtures
        Sample fixtures for testing.
    sample_player : FplPlayer
        Sample player for testing.
    """
    result = get_player_fixtures(sample_fixtures, sample_player)
    
    # Assertions
    assert isinstance(result, FplFixtures)
    assert len(result.fixtures) == 2  # Only unfinished fixtures for team 1
    
    # Verify we only have unfinished fixtures
    assert all(not fixture.finished for fixture in result.fixtures)
    
    # Verify all fixtures involve the player's team
    assert all(
        fixture.team_h == sample_player.team_id or fixture.team_a == sample_player.team_id
        for fixture in result.fixtures
    )
    
    # Check fixture IDs are as expected (2 and 3 from our sample data)
    fixture_ids = [fixture.id for fixture in result.fixtures]
    assert 2 in fixture_ids
    assert 3 in fixture_ids

def test_get_player_rolling_points(sample_rolling_data: pl.DataFrame, sample_player: FplPlayer) -> None:
    """Test the get_player_rolling_points function.
    
    Parameters
    ----------
    sample_rolling_data : pl.DataFrame
        Sample rolling data for testing.
    sample_player : FplPlayer
        Sample player for testing.
    """
    # Test with default season
    result = get_player_rolling_points(sample_rolling_data, sample_player)
    assert result == 6.5  # Most recent GW in 2024-25
    
    # Test with specific season
    result = get_player_rolling_points(sample_rolling_data, sample_player, season="2023-24")
    assert result == 5.8  # Most recent GW in 2023-24
    
    # Test with player not in data
    other_player = FplPlayer(
        id=999,
        first_name="Other",
        second_name="Player",
        web_name="O. Player",
        selected_by_percent=5.0,
        now_cost=50,
        team_id=2,
        element_type=2
    )
    result = get_player_rolling_points(sample_rolling_data, other_player)
    assert result == 0.0  # Player not found 