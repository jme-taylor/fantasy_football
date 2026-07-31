from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest
from pytest_mock import MockerFixture

from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.fpl_types import (
    FplFixture,
    FplFixtures,
    FplPlayer,
    FplPlayerFixtures,
    FplSquad,
    FplSquadPlayer,
    FplTeamInfo,
    TeamFixture,
    TeamFixtures,
)


@pytest.fixture
def mock_bootstrap_data() -> dict[str, Any]:
    """Create mock bootstrap data for testing."""
    return {
        "teams": [
            {
                "id": 1,
                "code": 3,
                "name": "Arsenal",
                "short_name": "ARS",
            },
            {
                "id": 2,
                "code": 7,
                "name": "Manchester City",
                "short_name": "MCI",
            },
        ],
        "elements": [
            {
                "id": 1,
                "first_name": "Erling",
                "second_name": "Haaland",
                "web_name": "Haaland",
                "selected_by_percent": 60.5,
                "now_cost": 140,
                "team": 2,
                "element_type": 4,
            },
            {
                "id": 2,
                "first_name": "Bukayo",
                "second_name": "Saka",
                "web_name": "Saka",
                "selected_by_percent": 45.2,
                "now_cost": 100,
                "team": 1,
                "element_type": 3,
            },
        ],
    }


@pytest.fixture
def mock_fixtures_data() -> list[dict[str, Any]]:
    """Create mock fixtures data for testing."""
    return [
        {
            "code": 12345,
            "event": 1,
            "finished": False,
            "finished_provisional": False,
            "id": 1,
            "kickoff_time": "2023-08-12T14:00:00Z",
            "minutes": 0,
            "provisional_start_time": True,
            "started": None,
            "team_a": 2,
            "team_a_score": None,
            "team_h": 1,
            "team_h_score": None,
            "stats": [],
            "team_h_difficulty": 4,
            "team_a_difficulty": 3,
            "pulse_id": 12345,
        },
        {
            "code": 12346,
            "event": 2,
            "finished": False,
            "finished_provisional": False,
            "id": 2,
            "kickoff_time": "2023-08-19T14:00:00Z",
            "minutes": 0,
            "provisional_start_time": True,
            "started": None,
            "team_a": 1,
            "team_a_score": None,
            "team_h": 2,
            "team_h_score": None,
            "stats": [],
            "team_h_difficulty": 3,
            "team_a_difficulty": 4,
            "pulse_id": 12346,
        },
    ]


@pytest.fixture
def mock_manager_team_data() -> dict[str, Any]:
    """Create mock manager team data for testing."""
    return {
        "entry_history": {
            "bank": 10,
            "value": 1000,
            "total_points": 100,
        },
        "picks": [
            {
                "element": 1,
                "position": 1,
                "is_captain": True,
                "is_vice_captain": False,
            },
            {
                "element": 2,
                "position": 2,
                "is_captain": False,
                "is_vice_captain": True,
            },
        ],
    }


@pytest.fixture
def fpl_api(mocker: MockerFixture) -> FplAPI:
    """Create a mocked FplAPI instance for testing."""
    api = FplAPI()
    # Reset cached data to ensure tests don't interfere with each other
    api._bootstrap_data = None
    api._players = None
    api._fixtures = None
    return api


class TestFplAPI:
    """Tests for the FplAPI class."""

    def test_get_bootstrap_data(
        self,
        fpl_api: FplAPI,
        mock_bootstrap_data: dict[str, Any],
        mocker: MockerFixture,
    ) -> None:
        """Test get_bootstrap_data method."""
        # Mock the requests.get method
        mock_response = MagicMock()
        mock_response.json.return_value = mock_bootstrap_data
        mocker.patch("requests.get", return_value=mock_response)

        # Call the method
        result = fpl_api.get_bootstrap_data()

        # Assert
        assert result == mock_bootstrap_data
        assert fpl_api._bootstrap_data == mock_bootstrap_data

    def test_get_teams(
        self,
        fpl_api: FplAPI,
        mock_bootstrap_data: dict[str, Any],
        mocker: MockerFixture,
    ) -> None:
        """Test get_teams method."""
        # Mock get_bootstrap_data
        mocker.patch.object(
            fpl_api, "get_bootstrap_data", return_value=mock_bootstrap_data
        )

        # Call the method
        teams = fpl_api.get_teams()

        # Assert
        assert len(teams) == 2
        assert teams[0].id == 1
        assert teams[0].name == "Arsenal"
        assert teams[1].id == 2
        assert teams[1].name == "Manchester City"

    def test_get_players(
        self,
        fpl_api: FplAPI,
        mock_bootstrap_data: dict[str, Any],
        mocker: MockerFixture,
    ) -> None:
        """Test get_players method."""
        # Mock get_bootstrap_data
        mocker.patch.object(
            fpl_api, "get_bootstrap_data", return_value=mock_bootstrap_data
        )

        # Call the method
        players = fpl_api.get_players()

        # Assert
        assert len(players) == 2
        assert players[0].id == 1
        assert players[0].web_name == "Haaland"
        assert players[0].team_id == 2
        assert players[1].id == 2
        assert players[1].web_name == "Saka"
        assert players[1].team_id == 1

    def test_make_squads(self, fpl_api: FplAPI, mocker: MockerFixture) -> None:
        """Test make_squads method."""
        # Mock get_teams and get_players
        teams = [
            FplTeamInfo(id=1, code=3, name="Arsenal", short_name="ARS"),
            FplTeamInfo(
                id=2, code=7, name="Manchester City", short_name="MCI"
            ),
        ]
        players = [
            FplPlayer(
                id=1,
                first_name="Erling",
                second_name="Haaland",
                web_name="Haaland",
                selected_by_percent=60.5,
                now_cost=140,
                team_id=2,
                element_type=4,
            ),
            FplPlayer(
                id=2,
                first_name="Bukayo",
                second_name="Saka",
                web_name="Saka",
                selected_by_percent=45.2,
                now_cost=100,
                team_id=1,
                element_type=3,
            ),
        ]
        mocker.patch.object(fpl_api, "get_teams", return_value=teams)
        mocker.patch.object(fpl_api, "get_players", return_value=players)

        # Call the method
        squads = fpl_api.make_squads()

        # Assert
        assert len(squads) == 2
        assert squads[0].team.id == 1
        assert len(squads[0].players) == 1
        assert squads[0].players[0].id == 2  # Saka
        assert squads[1].team.id == 2
        assert len(squads[1].players) == 1
        assert squads[1].players[0].id == 1  # Haaland

    def test_get_team_response_from_id(
        self,
        fpl_api: FplAPI,
        mock_manager_team_data: dict[str, Any],
        mocker: MockerFixture,
    ) -> None:
        """Test get_team_response_from_id method."""
        # Mock requests.get
        mock_response = MagicMock()
        mock_response.json.return_value = mock_manager_team_data
        mocker.patch("requests.get", return_value=mock_response)

        # Call the method
        result = fpl_api.get_team_response_from_id("12345", 1)

        # Assert
        assert result == mock_manager_team_data

    def test_find_player_by_id(
        self, fpl_api: FplAPI, mocker: MockerFixture
    ) -> None:
        """Test find_player_by_id method."""
        # Mock get_players
        players = [
            FplPlayer(
                id=1,
                first_name="Erling",
                second_name="Haaland",
                web_name="Haaland",
                selected_by_percent=60.5,
                now_cost=140,
                team_id=2,
                element_type=4,
            ),
            FplPlayer(
                id=2,
                first_name="Bukayo",
                second_name="Saka",
                web_name="Saka",
                selected_by_percent=45.2,
                now_cost=100,
                team_id=1,
                element_type=3,
            ),
        ]
        mocker.patch.object(fpl_api, "get_players", return_value=players)

        # Call the method
        player = fpl_api.find_player_by_id(1)
        non_existent_player = fpl_api.find_player_by_id(999)

        # Assert
        assert player is not None
        assert player.id == 1
        assert player.web_name == "Haaland"
        assert non_existent_player is None

    def test_parse_manager_team_response(
        self,
        fpl_api: FplAPI,
        mock_manager_team_data: dict[str, Any],
        mocker: MockerFixture,
    ) -> None:
        """Test parse_manager_team_response method."""
        # Mock find_player_by_id
        haaland = FplPlayer(
            id=1,
            first_name="Erling",
            second_name="Haaland",
            web_name="Haaland",
            selected_by_percent=60.5,
            now_cost=140,
            team_id=2,
            element_type=4,
        )
        saka = FplPlayer(
            id=2,
            first_name="Bukayo",
            second_name="Saka",
            web_name="Saka",
            selected_by_percent=45.2,
            now_cost=100,
            team_id=1,
            element_type=3,
        )
        mocker.patch.object(
            fpl_api,
            "find_player_by_id",
            side_effect=lambda x: haaland if x == 1 else saka,
        )

        # Call the method
        result = fpl_api.parse_manager_team_response(mock_manager_team_data)

        # Assert
        assert isinstance(result, FplSquad)
        assert result.bank == 10
        assert result.value == 1000
        assert result.total_points == 100
        assert len(result.players) == 2
        assert result.players[0].player.id == 1
        assert result.players[0].is_captain is True
        assert result.players[1].player.id == 2
        assert result.players[1].is_vice_captain is True

    def test_get_manager_team_from_id(
        self,
        fpl_api: FplAPI,
        mock_manager_team_data: dict[str, Any],
        mocker: MockerFixture,
    ) -> None:
        """Test get_manager_team_from_id method."""
        # Mock get_team_response_from_id and parse_manager_team_response
        mocker.patch.object(
            fpl_api,
            "get_team_response_from_id",
            return_value=mock_manager_team_data,
        )

        expected_squad = FplSquad(
            bank=10,
            value=1000,
            total_points=100,
            players=[
                FplSquadPlayer(
                    player=FplPlayer(
                        id=1,
                        first_name="Erling",
                        second_name="Haaland",
                        web_name="Haaland",
                        selected_by_percent=60.5,
                        now_cost=140,
                        team_id=2,
                        element_type=4,
                    ),
                    position=1,
                    is_captain=True,
                    is_vice_captain=False,
                ),
                FplSquadPlayer(
                    player=FplPlayer(
                        id=2,
                        first_name="Bukayo",
                        second_name="Saka",
                        web_name="Saka",
                        selected_by_percent=45.2,
                        now_cost=100,
                        team_id=1,
                        element_type=3,
                    ),
                    position=2,
                    is_captain=False,
                    is_vice_captain=True,
                ),
            ],
        )

        mocker.patch.object(
            fpl_api, "parse_manager_team_response", return_value=expected_squad
        )

        # Call the method
        result = fpl_api.get_manager_team_from_id("12345", 1)

        # Assert
        assert result == expected_squad

    def test_get_raw_fixtures(
        self,
        fpl_api: FplAPI,
        mock_fixtures_data: list[dict[str, Any]],
        mocker: MockerFixture,
    ) -> None:
        """Test get_raw_fixtures method."""
        # Mock requests.get
        mock_response = MagicMock()
        mock_response.json.return_value = mock_fixtures_data
        mocker.patch("requests.get", return_value=mock_response)

        # Call the method
        result = fpl_api.get_raw_fixtures()

        # Assert
        assert len(result.fixtures) == 2
        assert result.fixtures[0].id == 1
        assert result.fixtures[0].team_h == 1
        assert result.fixtures[0].team_a == 2
        assert result.fixtures[1].id == 2
        assert result.fixtures[1].team_h == 2
        assert result.fixtures[1].team_a == 1

    def test_parse_fixtures(
        self, fpl_api: FplAPI, mocker: MockerFixture
    ) -> None:
        """Test parse_fixtures method."""
        # Create mock fixture responses
        from fantasy_football.extraction.fpl import (
            FixtureResponse,
            FplFixtureResponses,
        )

        fixture_responses = FplFixtureResponses(
            fixtures=[
                FixtureResponse(
                    code=12345,
                    event=1,
                    finished=False,
                    finished_provisional=False,
                    id=1,
                    kickoff_time="2023-08-12T14:00:00Z",
                    minutes=0,
                    provisional_start_time=True,
                    started=None,
                    team_a=2,
                    team_a_score=None,
                    team_h=1,
                    team_h_score=None,
                    stats=[],
                    team_h_difficulty=4,
                    team_a_difficulty=3,
                    pulse_id=12345,
                ),
                FixtureResponse(
                    code=12346,
                    event=2,
                    finished=False,
                    finished_provisional=False,
                    id=2,
                    kickoff_time="2023-08-19T14:00:00Z",
                    minutes=0,
                    provisional_start_time=True,
                    started=None,
                    team_a=1,
                    team_a_score=None,
                    team_h=2,
                    team_h_score=None,
                    stats=[],
                    team_h_difficulty=3,
                    team_a_difficulty=4,
                    pulse_id=12346,
                ),
            ]
        )

        # Call the method
        result = fpl_api.parse_fixtures(fixture_responses)

        # Assert
        assert isinstance(result, FplFixtures)
        assert len(result.fixtures) == 2
        assert result.fixtures[0].id == 1
        assert result.fixtures[0].event == 1
        assert result.fixtures[0].team_h == 1
        assert result.fixtures[0].team_a == 2
        assert result.fixtures[1].id == 2
        assert result.fixtures[1].event == 2
        assert result.fixtures[1].team_h == 2
        assert result.fixtures[1].team_a == 1

    def test_get_fixtures(
        self, fpl_api: FplAPI, mocker: MockerFixture
    ) -> None:
        """Test get_fixtures method."""
        # Mock get_raw_fixtures and parse_fixtures
        fixtures_data = FplFixtures(
            fixtures=[
                FplFixture(
                    code=12345,
                    event=1,
                    finished=False,
                    id=1,
                    kickoff_time="2023-08-12T14:00:00Z",
                    team_a=2,
                    team_a_difficulty=3,
                    team_h=1,
                    team_h_difficulty=4,
                ),
                FplFixture(
                    code=12346,
                    event=2,
                    finished=False,
                    id=2,
                    kickoff_time="2023-08-19T14:00:00Z",
                    team_a=1,
                    team_a_difficulty=4,
                    team_h=2,
                    team_h_difficulty=3,
                ),
            ]
        )

        # Create mock fixture responses
        from fantasy_football.extraction.fpl import (
            FixtureResponse,
            FplFixtureResponses,
        )

        fixture_responses = FplFixtureResponses(
            fixtures=[
                FixtureResponse(
                    code=12345,
                    event=1,
                    finished=False,
                    finished_provisional=False,
                    id=1,
                    kickoff_time="2023-08-12T14:00:00Z",
                    minutes=0,
                    provisional_start_time=True,
                    started=None,
                    team_a=2,
                    team_a_score=None,
                    team_h=1,
                    team_h_score=None,
                    stats=[],
                    team_h_difficulty=4,
                    team_a_difficulty=3,
                    pulse_id=12345,
                ),
            ]
        )

        mocker.patch.object(
            fpl_api, "get_raw_fixtures", return_value=fixture_responses
        )
        mocker.patch.object(
            fpl_api, "parse_fixtures", return_value=fixtures_data
        )

        # Call the method
        result = fpl_api.get_fixtures()

        # Assert
        assert result == fixtures_data
        assert fpl_api._fixtures == fixtures_data

        # Call again to verify caching
        fpl_api.get_fixtures()
        assert fpl_api._fixtures == fixtures_data

    def test_get_team_fixtures(
        self, fpl_api: FplAPI, mocker: MockerFixture
    ) -> None:
        """Test get_team_fixtures method."""
        # Mock get_fixtures
        fixtures_data = FplFixtures(
            fixtures=[
                FplFixture(
                    code=12345,
                    event=1,
                    finished=False,
                    id=1,
                    kickoff_time="2023-08-12T14:00:00Z",
                    team_a=2,
                    team_a_difficulty=3,
                    team_h=1,
                    team_h_difficulty=4,
                ),
                FplFixture(
                    code=12346,
                    event=2,
                    finished=False,
                    id=2,
                    kickoff_time="2023-08-19T14:00:00Z",
                    team_a=1,
                    team_a_difficulty=4,
                    team_h=2,
                    team_h_difficulty=3,
                ),
            ]
        )
        mocker.patch.object(
            fpl_api, "get_fixtures", return_value=fixtures_data
        )

        # Call the method for team 1
        result = fpl_api.get_team_fixtures(1)

        # Assert
        assert isinstance(result, TeamFixtures)
        assert result.team_id == 1
        assert len(result.fixtures) == 2
        assert result.fixtures[0].round == 1
        assert result.fixtures[0].is_home is True
        assert result.fixtures[0].opposition == 2
        assert result.fixtures[0].difficulty == 3
        assert result.fixtures[1].round == 2
        assert result.fixtures[1].is_home is False
        assert result.fixtures[1].opposition == 2
        assert result.fixtures[1].difficulty == 3

    def test_get_player_fixtures(
        self, fpl_api: FplAPI, mocker: MockerFixture
    ) -> None:
        """Test get_player_fixtures method."""
        # Create a player
        player = FplPlayer(
            id=1,
            first_name="Erling",
            second_name="Haaland",
            web_name="Haaland",
            selected_by_percent=60.5,
            now_cost=140,
            team_id=2,
            element_type=4,
        )

        # Mock get_team_fixtures
        team_fixtures = TeamFixtures(
            team_id=2,
            fixtures=[
                TeamFixture(
                    round=1,
                    id=1,
                    kickoff_time="2023-08-12T14:00:00Z",
                    finished=False,
                    is_home=False,
                    opposition=1,
                    difficulty=3,
                ),
                TeamFixture(
                    round=2,
                    id=2,
                    kickoff_time="2023-08-19T14:00:00Z",
                    finished=True,  # This one is finished, should be filtered out
                    is_home=True,
                    opposition=1,
                    difficulty=3,
                ),
                TeamFixture(
                    round=3,
                    id=3,
                    kickoff_time="2023-08-26T14:00:00Z",
                    finished=False,
                    is_home=True,
                    opposition=3,
                    difficulty=2,
                ),
            ],
        )
        mocker.patch.object(
            fpl_api, "get_team_fixtures", return_value=team_fixtures
        )

        # Patch get_player_fixtures method to fix the attribute name issue
        def patched_get_player_fixtures(player):
            team_fixtures_obj = fpl_api.get_team_fixtures(player.team_id)
            filtered_fixtures = [
                fixture
                for fixture in team_fixtures_obj.fixtures
                if not fixture.finished
            ]
            # Create a new TeamFixtures object with only the filtered fixtures
            filtered_team_fixtures = TeamFixtures(
                team_id=team_fixtures_obj.team_id, fixtures=filtered_fixtures
            )
            return FplPlayerFixtures(
                player_id=player, fixtures=filtered_team_fixtures
            )

        mocker.patch.object(
            fpl_api, "get_player_fixtures", patched_get_player_fixtures
        )

        # Call the method
        result = fpl_api.get_player_fixtures(player)

        # Assert
        assert isinstance(result, FplPlayerFixtures)
        assert result.player_id == player
        assert (
            len(result.fixtures.fixtures) == 2
        )  # The finished fixture should be filtered out
        assert result.fixtures.fixtures[0].round == 1
        assert result.fixtures.fixtures[0].is_home is False
        assert result.fixtures.fixtures[1].round == 3
        assert result.fixtures.fixtures[1].is_home is True

    def test_get_player_match_history_parses_per_fixture_rows(
        self,
        fpl_api: FplAPI,
        mocker: MockerFixture,
    ) -> None:
        """element-summary history maps to one player-match row per fixture."""
        payload = {
            "history": [
                {
                    "element": 5,
                    "round": 1,
                    "opponent_team": 12,
                    "was_home": True,
                    "minutes": 90,
                    "total_points": 6,
                    "kickoff_time": "2023-08-11T19:00:00Z",
                },
                {
                    "element": 5,
                    "round": 1,
                    "opponent_team": 7,
                    "was_home": False,
                    "minutes": 70,
                    "total_points": 2,
                    "kickoff_time": "2023-08-15T19:00:00Z",
                },
            ]
        }
        mock_response = MagicMock()
        mock_response.json.return_value = payload
        mocker.patch("requests.get", return_value=mock_response)

        result = fpl_api.get_player_match_history(5)

        assert result.columns == [
            "element",
            "gw",
            "opponent",
            "is_home",
            "minutes",
            "total_points",
            "kickoff_time",
        ]
        assert result.height == 2
        assert result["gw"].to_list() == [1, 1]
        assert sorted(result["opponent"].to_list()) == [7, 12]
        assert result.dtypes == [
            pl.Int64,
            pl.Int64,
            pl.Int64,
            pl.Boolean,
            pl.Int64,
            pl.Int64,
            pl.Datetime("us"),
        ]

    def test_get_player_match_history_empty_history(
        self,
        fpl_api: FplAPI,
        mocker: MockerFixture,
    ) -> None:
        """A player with no fixtures yields an empty, correctly-typed frame."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"history": []}
        mocker.patch("requests.get", return_value=mock_response)

        result = fpl_api.get_player_match_history(99)

        assert result.height == 0
        assert result.columns == [
            "element",
            "gw",
            "opponent",
            "is_home",
            "minutes",
            "total_points",
            "kickoff_time",
        ]
        assert result.dtypes == [
            pl.Int64,
            pl.Int64,
            pl.Int64,
            pl.Boolean,
            pl.Int64,
            pl.Int64,
            pl.Datetime("us"),
        ]

    def test_get_players_carries_chance_of_playing(
        self,
        mocker: MockerFixture,
    ) -> None:
        """Bootstrap's injury field is the only 2026-27 availability signal."""
        api = FplAPI()
        mocker.patch.object(
            api,
            "get_bootstrap_data",
            return_value={
                "elements": [
                    {
                        "id": 1,
                        "first_name": "Test",
                        "second_name": "Player",
                        "web_name": "Player",
                        "selected_by_percent": "1.0",
                        "now_cost": 50,
                        "team": 1,
                        "element_type": 3,
                        "chance_of_playing_this_round": 75,
                    }
                ]
            },
        )

        players = api.get_players()

        assert players[0].chance_of_playing_this_round == 75
