from typing import Iterator

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

config = ConfigDict(extra="forbid")


@dataclass(config=config, frozen=True)
class FplTeamInfo:
    """Class for storing team information.

    Attributes
    ----------
    id : int
        The team's unique identifier
    code : int
        The team's code
    name : str
        The team's full name
    short_name : str
        The team's short name

    """

    id: int
    code: int
    name: str
    short_name: str


@dataclass(config=config, frozen=True)
class FplPlayer:
    """Class for storing player information.

    Attributes
    ----------
    id : int
        The player's unique identifier
    first_name : str
        The player's first name
    second_name : str
        The player's second name
    web_name : str
        How the player's name appears on the FPL website
    selected_by_percent : float
        The percentage of players who have selected the player
    now_cost : int
        The player's current cost
    team_id : int
        The ID of the team the player plays for
    element_type : int
        The player's position ID.
    """

    id: int
    first_name: str
    second_name: str
    web_name: str
    selected_by_percent: float
    now_cost: int
    team_id: int
    element_type: int


@dataclass(config=config, frozen=True)
class FplTeam:
    """Class for storing team information.

    Attributes
    ----------
    team : FplTeamInfo
        The team's information
    players : list[FplPlayer]
        All the players in the team
    """

    team: FplTeamInfo
    players: list[FplPlayer]


@dataclass(config=config, frozen=True)
class FplSquadPlayer:
    """Class for storing player information in a squad.

    Attributes
    ----------
    player : FplPlayer
        The player object
    position : int
        The player's position in the squad
    is_captain : bool
        Whether the player is the captain
    is_vice_captain : bool
        Whether the player is the vice captain
    """

    player: FplPlayer
    position: int
    is_captain: bool
    is_vice_captain: bool


@dataclass(config=config, frozen=True)
class FplSquad:
    """Class for storing an FPL squad.

    Attributes
    ----------
    bank : int
        The amount of money in the bank
    value : int
        The total value of the squad
    total_points : int
        The total points the squad has scored
    players : list[FplSquadPlayer]
        The players in the squad currently.
    """

    bank: int
    value: int
    total_points: int
    players: list[FplSquadPlayer]


@dataclass(config=config, frozen=True)
class FplFixtureInfo:
    """Class for storing fixture information.

    Attributes
    ----------
    event : int
        Which gameweek the fixture is in.
    finished : bool
        Whether the fixture has finished.
    home_team : int
        The ID of the home team.
    home_team_fixture_difficulty : int
        The fixture difficulty for the home team.
    away_team : int
        The ID of the away team.
    away_team_fixture_difficulty : int
        The fixture difficulty for the away team.
    """

    event: int
    finished: bool
    home_team: int
    home_team_fixture_difficulty: int
    away_team: int
    away_team_fixture_difficulty: int


@dataclass(config=config, frozen=True)
class FplTeamFixture:
    """Class for storing fixture information from the perspective of a team.

    Attributes
    ----------
    event: int
        Which gameweek the fixture is in.
    finished: bool
        Whether the fixture has finished.
    is_home: bool
        Whether the team is playing at home.
    kickoff_time: str
        The time the fixture kicks off
    """

    event: int
    finished: bool
    is_home: bool
    kickoff_time: str


@dataclass(config=config, frozen=True)
class FplFixture:
    """Class for stroing fixture information from FPL.

    Attributes
    ----------
    code : int
        The fixture's code
    event : int
        The gameweek the fixture is in
    finished : bool
        Whether the fixture has finished
    id : int
        The fixture's unique identifier
    kickoff_time : str
        The time the fixture kicks off
    team_a : int
        The ID of the away team
    team_a_difficulty : int
        The fixture difficulty for the away team
    team_h : int
        The ID of the home team
    team_h_difficulty : int
        The fixture difficulty for the home team
    """

    code: int
    event: int
    finished: bool
    id: int
    kickoff_time: str
    team_a: int
    team_a_difficulty: int
    team_h: int
    team_h_difficulty: int


@dataclass(config=config, frozen=True)
class FplFixtures:
    """Class for holding a list of fixtures.

    Attributes
    ----------
    fixtures : list[FplFixture]
        A list of fixtures
    """

    fixtures: list[FplFixture]

    def __iter__(self) -> Iterator[FplFixture]:
        """Iterate over the fixtures.

        Returns
        -------
        Iterator[FplFixture]
            An iterator over the fixtures.
        """
        return iter(self.fixtures)


@dataclass(config=config, frozen=True)
class TeamFixture:
    """Class for storing fixture information from the perspective of a team.

    Attributes
    ----------
    round : int
        The gameweek the fixture is in
    id : int
        The fixture's unique identifier
    kickoff_time : str
        The time the fixture kicks off
    finished : bool
        Whether the fixture has finished
    is_home : bool
        Whether the team is playing at home
    opposition : int
        The ID of the opposition team
    difficulty : int
        The fixture difficulty for the team
    """

    round: int
    id: int
    kickoff_time: str
    finished: bool
    is_home: bool
    opposition: int
    difficulty: int


@dataclass(config=config, frozen=True)
class TeamFixtures:
    """Class for storing a list of fixtures from the perspective of a team.

    Attributes
    ----------
    team_id : int
        The ID of the team
    fixtures : list[TeamFixture]
        A list of fixtures
    """

    team_id: int
    fixtures: list[TeamFixture]


@dataclass(config=config, frozen=True)
class FplPlayerFixtures:
    """Class for storing a player's fixtures.

    Attributes
    ----------
    player_id : FplPlayer
        The player's information
    fixtures : TeamFixtures
        A list of fixtures for their team
    """

    player_id: FplPlayer
    fixtures: TeamFixtures


@dataclass(config=config, frozen=True)
class PlayerExpectedPoints:
    """Class for storing a player's expected points.

    Attributes
    ----------
    player : FplPlayer
        The player's information
    fixture : TeamFixture
        The fixture the player is playing in
    rolling_points : float
        The player's rolling points average
    expected_points : float
        The player's expected points for the fixture
    """

    player: FplPlayer
    fixture: TeamFixture
    rolling_points: float
    expected_points: float


@dataclass(config=config, frozen=True)
class PlayerGameweekExpectedPoints:
    """Class for storing a player's expected points over a gameweek.

    Attributes
    ----------
    player_id: int
        The id of the player in question, from the FPL API.
    player_name: str
        The player's name.
    expected_points: float
        How many points we expect them to get in the gameweek.
    """

    player_id: int
    player_name: str
    expected_points: float


@dataclass(config=config, frozen=True)
class GameWeekPlan:
    """Class for the output of a gameweek plan.

    Attributes
    ----------
    gameweek: int
        The gameweek in question
    squad: list[PlayerGameweekExpectedPoints]
        A list of our whole squad with expected points in the gameweek
    starting_xi: list[PlayerGameweekExpectedPoints]
        A list of the starting xi with expected points for the gameweek
    captain: PlayerGameweekExpectedPoints
        The player selected as captain for the gameweek
    transfers_in: list[PlayerGameweekExpectedPoints]
        A list of the players we want to transfer in this gameweek.
    transfers_out: list[PlayerGameweekExpectedPoints]
        A list of the players we want to transfer out this gameweek.
    hits: int
        The amount of transfer hits we took.
    free_transfers: int
        The amount of free transfers we'll have left this gameweek.
    expected_points: float
        How many points we expect this gameweek.
    """

    gameweek: int
    squad: list[PlayerGameweekExpectedPoints]
    starting_xi: list[PlayerGameweekExpectedPoints]
    captain: PlayerGameweekExpectedPoints
    transfers_in: list[PlayerGameweekExpectedPoints]
    transfers_out: list[PlayerGameweekExpectedPoints]
    hits: int
    free_transfers: int
    expected_points: float
