
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


def get_boostrap_data() -> dict:
    """Get the bootstrap data from the FPL API.

    Returns
    -------
    dict
        The bootstrap data from the FPL API.
    """
    base_url = "https://fantasy.premierleague.com/api/"
    response_json = requests.get(base_url + "bootstrap-static/").json()
    return response_json


def get_teams(data: dict) -> list[FplTeamInfo]:
    """Get the teams from the bootstrap data.

    Parameters
    ----------
    data : dict
        The bootstrap data from the FPL API.

    Returns
    -------
    list[FplTeamInfo]
        A list of all teams from the FPL API.
    """
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


def get_players() -> list[FplPlayer]:
    """Get all players from the FPL API.

    This function gets all players from the FPL API and returns a list of
    FplPlayer objects. This is used to get the player data for the current
    season.

    Returns
    -------
    list[FplPlayer]
        A list of all players from the FPL API.
    """
    bootstrap_dict = get_boostrap_data()
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
        )
        players.append(fpl_player)
    return players


def make_squads(
    teams: list[FplTeam], players: list[FplPlayer]
) -> list[FplTeam]:
    """Make a list of FplTeam objects from a list of FplPlayer objects.

    Parameters
    ----------
    teams : list[FplTeam]
        A list of all teams from the FPL API.
    players : list[FplPlayer]
        A list of all players from the FPL API.

    Returns
    -------
    list[FplTeam]
        A list of all teams from the FPL API.
    """
    squads = []
    for team in teams:
        team_players = [
            player for player in players if player.team_id == team.id
        ]
        squad = FplTeam(team=team, players=team_players)
        squads.append(squad)
    return squads


def get_team_response_from_id(id: str, event: int) -> dict:
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
    url = f"https://fantasy.premierleague.com/api/entry/{id}/event/{event}/picks/"
    response = requests.get(url)
    response.raise_for_status()
    return response.json()


def find_player_by_id(
    players: list[FplPlayer], player_id: int
) -> FplPlayer | None:
    """Find a player by their ID.

    Parameters
    ----------
    players : list[FplPlayer]
        A list of all players from the FPL API.
    player_id : int
        The ID of the player to find.

    Returns
    -------
    FplPlayer | None
        The player object if found, otherwise None.
    """
    for player in players:
        if player.id == player_id:
            return player
    return None


def parse_manager_team_response(
    response: dict, players: list[FplPlayer]
) -> FplSquad:
    """Parse the manager team response from the FPL API.

    Parameters
    ----------
    response : dict
        The manager team response from the FPL API.
    players : list[FplPlayer]
        A list of all players from the FPL API.

    Returns
    -------
    FplSquad
        The manager team from the FPL API.
    """
    bank = response.get("entry_history").get("bank")
    value = response.get("entry_history").get("value")
    total_points = response.get("entry_history").get("total_points")
    squad_players = response.get("picks")
    manager_squad_players = []
    for squad_player in squad_players:
        player_id = squad_player.get("element")
        position = squad_player.get("position")
        is_captain = squad_player.get("is_captain")
        is_vice_captain = squad_player.get("is_vice_captain")
        player = find_player_by_id(players, player_id)
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


def get_manager_team_from_id(
    id: str, event: int, players: list[FplPlayer]
) -> FplSquad:
    """Get and parse the manager team response from the FPL API.

    Parameters
    ----------
    id : str
        The ID of the manager to get the team for.
    event : int
        The gameweek to get the team for.
    players : list[FplPlayer]
        A list of all players from the FPL API.

    Returns
    -------
    FplSquad
        The manager team from the FPL API.
    """
    manager_response = get_team_response_from_id(id, event)
    manager_squad = parse_manager_team_response(manager_response, players)
    return manager_squad


def get_raw_fixtures() -> FplFixtureResponses:
    """Get the raw fixtures from the FPL API.

    Returns
    -------
    FplFixtureResponses
        The raw fixtures from the FPL API.
    """
    url = "https://fantasy.premierleague.com/api/fixtures/"
    response = requests.get(url)
    response.raise_for_status()
    all_fixtures = []
    all_fixtures_response = response.json()
    for fixture in all_fixtures_response:
        fixture_response = FixtureResponse(**fixture)
        all_fixtures.append(fixture_response)
    return FplFixtureResponses(fixtures=all_fixtures)


def parse_fixtures(fixtures: FplFixtureResponses) -> FplFixtures:
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
        if fixture.event is None:
            continue
        fpl_fixture = FplFixture(
            code=fixture.code,
            event=fixture.event,
            finished=fixture.finished,
            id=fixture.id,
            kickoff_time=fixture.kickoff_time,
            team_a=fixture.team_a,
            team_a_difficulty=fixture.team_a_difficulty,
            team_h=fixture.team_h,
            team_h_difficulty=fixture.team_h_difficulty,
        )
        all_fixtures.append(fpl_fixture)
    return FplFixtures(fixtures=all_fixtures)


def get_fixtures() -> FplFixtures:
    """Get and parse the fixtures from the FPL API.

    Returns
    -------
    FplFixtures
        The parsed fixtures from the FPL API.
    """
    raw_fixtures = get_raw_fixtures()
    return parse_fixtures(raw_fixtures)


def get_team_fixtures(team_id: int, fixtures: FplFixtures) -> TeamFixtures:
    """Get the team fixtures from the FPL API.

    Parameters
    ----------
    team_id : int
        _description_
    fixtures : FplFixtures
        _description_

    Returns
    -------
    TeamFixtures
        _description_
    """
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


def get_player_fixtures(
    player: FplPlayer, fixtures: FplFixtures
) -> FplPlayerFixtures:
    """Get the player fixtures from the FPL API.

    Parameters
    ----------
    player : FplPlayer
        The player to get the fixtures for.
    fixtures : FplFixtures
        The parsed fixtures from the FPL API.

    Returns
    -------
    FplPlayerFixtures
        The player fixtures from the FPL API.
    """
    team_fixtures = get_team_fixtures(player.team_id, fixtures)
    player_fixtures = []
    for fixture in team_fixtures.fixtures:
        if not fixture.is_finished:
            player_fixtures.append(fixture)

    return FplPlayerFixtures(player_id=player, fixtures=player_fixtures)
