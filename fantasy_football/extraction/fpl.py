import polars as pl
import requests
from pydantic import BaseModel

from fantasy_football.fpl_types import (
    FplFixture,
    FplFixtures,
    FplPlayer,
    FplPlayerFixtures,
    FplSquad,
    FplSquadPlayer,
    FplTeam,
    FplTeamInfo,
    TeamFixture,
    TeamFixtures,
)

PLAYER_MATCH_HISTORY_COLUMNS: list[str] = [
    "element",
    "gw",
    "opponent",
    "is_home",
    "minutes",
    "total_points",
    "yellow_cards",
    "red_cards",
    "kickoff_time",
]


class FixtureResponse(BaseModel):
    """Class for storing a fixture response from the FPL API."""

    class FixtureStatistic(BaseModel):  # noqa : DCO1
        class TeamStatistic(BaseModel):  # noqa : DCO1
            value: int | None = None
            element: int | None = None

        identifier: str
        a: list[TeamStatistic]
        h: list[TeamStatistic]

    code: int
    event: int | None = None
    finished: bool
    finished_provisional: bool
    id: int
    kickoff_time: str | None = None
    minutes: int
    provisional_start_time: bool
    started: bool | None = None
    team_a: int
    team_a_score: int | None = None
    team_h: int
    team_h_score: int | None = None
    stats: list[FixtureStatistic]
    team_h_difficulty: int
    team_a_difficulty: int
    pulse_id: int


class FplFixtureResponses(BaseModel):  # noqa : DCO1
    fixtures: list[FixtureResponse]


class FplAPI:
    """Class for interacting with the Fantasy Premier League API."""

    BASE_URL = "https://fantasy.premierleague.com/api/"

    def __init__(self):
        """Initialize the FplAPI class."""
        self._bootstrap_data = None
        self._players = None
        self._fixtures = None

    def get_bootstrap_data(self) -> dict:
        """Get the bootstrap data from the FPL API.

        Returns
        -------
        dict
            The bootstrap data from the FPL API.
        """
        if self._bootstrap_data is None:
            response_json = requests.get(
                self.BASE_URL + "bootstrap-static/"
            ).json()
            self._bootstrap_data = response_json
        return self._bootstrap_data

    def get_teams(self) -> list[FplTeamInfo]:
        """Get the teams from the bootstrap data.

        Returns
        -------
        list[FplTeamInfo]
            A list of all teams from the FPL API.
        """
        data = self.get_bootstrap_data()
        teams_list = data.get("teams")
        teams = []
        for team in teams_list:
            fpl_team = FplTeamInfo(
                id=team.get("id"),
                code=team.get("code"),
                name=team.get("name"),
                short_name=team.get("short_name"),
            )
            teams.append(fpl_team)
        return teams

    def get_players(self) -> list[FplPlayer]:
        """Get all players from the FPL API.

        This function gets all players from the FPL API and returns a list of
        FplPlayer objects. This is used to get the player data for the current
        season.

        Returns
        -------
        list[FplPlayer]
            A list of all players from the FPL API.
        """
        if self._players is None:
            bootstrap_dict = self.get_bootstrap_data()
            elements = bootstrap_dict.get("elements")
            players = []
            for player in elements:
                fpl_player = FplPlayer(
                    id=player.get("id"),
                    first_name=player.get("first_name"),
                    second_name=player.get("second_name"),
                    web_name=player.get("web_name"),
                    selected_by_percent=player.get("selected_by_percent"),
                    now_cost=player.get("now_cost"),
                    team_id=player.get("team"),
                    element_type=player.get("element_type"),
                    chance_of_playing_this_round=player.get(
                        "chance_of_playing_this_round"
                    ),
                )
                players.append(fpl_player)
            self._players = players
        return self._players

    def make_squads(self) -> list[FplTeam]:
        """Make a list of FplTeam objects from a list of FplPlayer objects.

        Returns
        -------
        list[FplTeam]
            A list of all teams from the FPL API with their players.
        """
        teams = self.get_teams()
        players = self.get_players()
        squads = []
        for team in teams:
            team_players = [
                player for player in players if player.team_id == team.id
            ]
            squad = FplTeam(team=team, players=team_players)
            squads.append(squad)
        return squads

    def get_team_response_from_id(self, id: str, event: int) -> dict:
        """Get the team response from the FPL API.

        Parameters
        ----------
        id : str
            The ID of the team to get the response for.
        event : int
            The gameweek to get the response for.

        Returns
        -------
        dict
            The team response from the FPL API.
        """
        url = f"{self.BASE_URL}entry/{id}/event/{event}/picks/"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()

    def find_player_by_id(self, player_id: int) -> FplPlayer | None:
        """Find a player by their ID.

        Parameters
        ----------
        player_id : int
            The ID of the player to find.

        Returns
        -------
        FplPlayer | None
            The player object if found, otherwise None.
        """
        players = self.get_players()
        for player in players:
            if player.id == player_id:
                return player
        return None

    def parse_manager_team_response(self, response: dict) -> FplSquad:
        """Parse the manager team response from the FPL API.

        Parameters
        ----------
        response : dict
            The manager team response from the FPL API.

        Returns
        -------
        FplSquad
            The manager team from the FPL API.
        """
        entry_history = response["entry_history"]
        bank = entry_history.get("bank")
        value = entry_history.get("value")
        total_points = entry_history.get("total_points")
        squad_players = response["picks"]
        manager_squad_players = []
        for squad_player in squad_players:
            player_id = squad_player.get("element")
            position = squad_player.get("position")
            is_captain = squad_player.get("is_captain")
            is_vice_captain = squad_player.get("is_vice_captain")
            player = self.find_player_by_id(player_id)
            squad_player = FplSquadPlayer(
                player=player,
                position=position,
                is_captain=is_captain,
                is_vice_captain=is_vice_captain,
            )
            manager_squad_players.append(squad_player)
        return FplSquad(
            bank=bank,
            value=value,
            total_points=total_points,
            players=manager_squad_players,
        )

    def get_manager_team_from_id(self, id: str, event: int) -> FplSquad:
        """Get and parse the manager team response from the FPL API.

        Parameters
        ----------
        id : str
            The ID of the manager to get the team for.
        event : int
            The gameweek to get the team for.

        Returns
        -------
        FplSquad
            The manager team from the FPL API.
        """
        manager_response = self.get_team_response_from_id(id, event)
        manager_squad = self.parse_manager_team_response(manager_response)
        return manager_squad

    def get_raw_fixtures(self) -> FplFixtureResponses:
        """Get the raw fixtures from the FPL API.

        Returns
        -------
        FplFixtureResponses
            The raw fixtures from the FPL API.
        """
        url = f"{self.BASE_URL}fixtures/"
        response = requests.get(url)
        response.raise_for_status()
        all_fixtures = []
        all_fixtures_response = response.json()
        for fixture in all_fixtures_response:
            fixture_response = FixtureResponse.model_validate(fixture)
            all_fixtures.append(fixture_response)
        return FplFixtureResponses(fixtures=all_fixtures)

    def parse_fixtures(self, fixtures: FplFixtureResponses) -> FplFixtures:
        """Parse the raw fixtures from the FPL API.

        Parameters
        ----------
        fixtures : FplFixtureResponses
            The raw fixtures from the FPL API.

        Returns
        -------
        FplFixtures
            The parsed fixtures from the FPL API.
        """
        all_fixtures = []
        for fixture in fixtures.fixtures:
            event = fixture.event
            kickoff_time = fixture.kickoff_time
            if event is None or kickoff_time is None:
                continue
            fpl_fixture = FplFixture(
                code=fixture.code,
                event=event,
                finished=fixture.finished,
                id=fixture.id,
                kickoff_time=kickoff_time,
                team_a=fixture.team_a,
                team_a_difficulty=fixture.team_a_difficulty,
                team_h=fixture.team_h,
                team_h_difficulty=fixture.team_h_difficulty,
            )
            all_fixtures.append(fpl_fixture)
        return FplFixtures(fixtures=all_fixtures)

    def get_fixtures(self) -> FplFixtures:
        """Get and parse the fixtures from the FPL API.

        Returns
        -------
        FplFixtures
            The parsed fixtures from the FPL API.
        """
        if self._fixtures is None:
            raw_fixtures = self.get_raw_fixtures()
            self._fixtures = self.parse_fixtures(raw_fixtures)
        return self._fixtures

    def get_team_fixtures(self, team_id: int) -> TeamFixtures:
        """Get the team fixtures from the FPL API.

        Parameters
        ----------
        team_id : int
            The ID of the team to get fixtures for.

        Returns
        -------
        TeamFixtures
            The team fixtures from the FPL API.
        """
        fixtures = self.get_fixtures()
        team_fixtures = []
        for fixture in fixtures.fixtures:
            if fixture.team_a == team_id:
                is_home = False
                opposition = fixture.team_h
                difficulty = fixture.team_h_difficulty
            elif fixture.team_h == team_id:
                is_home = True
                opposition = fixture.team_a
                difficulty = fixture.team_a_difficulty
            else:
                continue
            team_fixture = TeamFixture(
                round=fixture.event,
                id=fixture.id,
                kickoff_time=fixture.kickoff_time,
                finished=fixture.finished,
                is_home=is_home,
                opposition=opposition,
                difficulty=difficulty,
            )
            team_fixtures.append(team_fixture)
        return TeamFixtures(team_id=team_id, fixtures=team_fixtures)

    def get_player_fixtures(self, player: FplPlayer) -> FplPlayerFixtures:
        """Get the player fixtures from the FPL API.

        Parameters
        ----------
        player : FplPlayer
            The player to get the fixtures for.

        Returns
        -------
        FplPlayerFixtures
            The player fixtures from the FPL API.
        """
        team_fixtures = self.get_team_fixtures(player.team_id)
        filtered_fixtures = [
            fixture
            for fixture in team_fixtures.fixtures
            if not fixture.finished
        ]
        # Create a new TeamFixtures object with only non-finished fixtures
        player_team_fixtures = TeamFixtures(
            team_id=team_fixtures.team_id, fixtures=filtered_fixtures
        )
        return FplPlayerFixtures(
            player_id=player, fixtures=player_team_fixtures
        )

    def get_player_match_history(self, element_id: int) -> pl.DataFrame:
        """Return a player's current-season per-fixture history.

        Hits ``element-summary/{element_id}/`` and reshapes its ``history``
        array (one entry per fixture) into canonical player-match columns. The
        endpoint only holds the current season — past seasons collapse to
        season aggregates upstream and are not returned here.

        Parameters
        ----------
        element_id : int
            The FPL element id.

        Returns
        -------
        pl.DataFrame
            Columns ``element, gw, opponent, is_home, minutes, total_points,
            yellow_cards, red_cards, kickoff_time``; empty (with that schema)
            when the player has no fixtures.
        """
        url = f"{self.BASE_URL}element-summary/{element_id}/"
        history = requests.get(url).json()["history"]
        empty_schema = dict(
            zip(
                PLAYER_MATCH_HISTORY_COLUMNS,
                [
                    pl.Int64,
                    pl.Int64,
                    pl.Int64,
                    pl.Boolean,
                    pl.Int64,
                    pl.Int64,
                    pl.Int64,
                    pl.Int64,
                    pl.Datetime("us"),
                ],
            )
        )
        if not history:
            return pl.DataFrame(schema=empty_schema)
        frame = pl.DataFrame(history).select(
            pl.col("element").cast(pl.Int64),
            pl.col("round").cast(pl.Int64).alias("gw"),
            pl.col("opponent_team").cast(pl.Int64).alias("opponent"),
            pl.col("was_home").alias("is_home"),
            pl.col("minutes").cast(pl.Int64),
            pl.col("total_points").cast(pl.Int64),
            pl.col("yellow_cards").cast(pl.Int64),
            pl.col("red_cards").cast(pl.Int64),
            # strict=False so a blank or malformed timestamp becomes a
            # null rather than aborting the whole load.
            pl.col("kickoff_time")
            .str.replace("Z", "+00:00")
            .str.to_datetime(time_zone="UTC", strict=False)
            .dt.replace_time_zone(None)
            .alias("kickoff_time"),
        )
        assert frame.columns == PLAYER_MATCH_HISTORY_COLUMNS
        return frame
