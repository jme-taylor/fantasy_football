from typing import Iterator

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

config = ConfigDict(extra="forbid")

@dataclass(config=config, frozen=True)
class FplTeamInfo:
    """Class for storing team information"""
    id: int
    code: int
    name: str
    short_name: str

@dataclass(config=config, frozen=True)
class FplPlayer:
    """Class for storing player information"""
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
    """Class for storing team information."""

    team: FplTeamInfo
    players: list[FplPlayer]

@dataclass(config=config, frozen=True)
class FplSquadPlayer:
    """Class for storing squad player information"""
    player: FplPlayer
    position: int
    is_captain: bool
    is_vice_captain: bool

@dataclass(config=config, frozen=True)
class FplSquad:
    bank: int
    value: int
    total_points: int
    players: list[FplSquadPlayer]

@dataclass(config=config, frozen=True)
class FplFixtureInfo:
    event: int
    finished: bool
    home_team: int
    home_team_fixture_difficulty: int
    away_team: int
    away_team_fixture_difficulty: int

@dataclass(config=config, frozen=True)
class FplTeamFixture:
    event: int
    finished: bool
    is_home: bool
    kickoff_time: str
    
@dataclass(config=config, frozen=True)
class FplFixture:
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
    fixtures: list[FplFixture]

    def __iter__(self) -> Iterator[FplFixture]:
        return iter(self.fixtures)

@dataclass(config=config, frozen=True)
class TeamFixture:
    round: int
    id: int
    kickoff_time: str
    finished: bool
    is_home: bool
    opposition: int
    difficulty: int

@dataclass(config=config, frozen=True)
class TeamFixtures:
    team_id: int
    fixtures: list[TeamFixture]

@dataclass(config=config, frozen=True)
class FplPlayerFixtures:
    player_id: FplPlayer
    fixtures: TeamFixtures

@dataclass(config=config, frozen=True)
class PlayerExpectedPoints:
    player: FplPlayer
    fixture: TeamFixture
    rolling_points: float
    expected_points: float 