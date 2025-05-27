import polars as pl

from fantasy_football.constants import DATA_FOLDER
from fantasy_football.fpl import FplAPI
from fantasy_football.fpl_types import (
    FplFixtures,
    FplPlayer,
    PlayerExpectedPoints,
    TeamFixture,
)


def get_player_rolling_points(
    rolling_data: pl.DataFrame, player: FplPlayer, season: str = "2024-25"
) -> float:
    """Get a players rolling points for a given gameweek.

    Parameters
    ----------
    rolling_data : pl.DataFrame
        The rolling data to use for predictions.
    player : FplPlayer
        The player to predict points for.
    season : str, optional
        The season to filter data for, by default "2024-25"

    Returns
    -------
    float
        Predicted points based on 5-game rolling average

    """
    season_data = rolling_data.filter(pl.col("season") == season)
    player_data = season_data.filter(pl.col("element") == player.id).sort(
        "gw", descending=True
    )

    if len(player_data) == 0:
        return 0.0

    return player_data.select("total_points_rolling_5")[0, 0]


def get_player_fixtures(
    fixtures: FplFixtures, player: FplPlayer
) -> list[TeamFixture]:
    """Get a player's upcoming fixtures.

    This function will return a list of upcoming fixtures for a given player.
    It will filter the fixtures to only ones that haven't occured yet.

    Parameters
    ----------
    fixtures : FplFixtures
        The fixtures to use for predictions.
    player : FplPlayer
        The player to predict fixtures for.

    Returns
    -------
    list[TeamFixture]
        A list of upcoming fixtures for the player.
    """
    team_id = player.team_id
    unfinished_fixtures = [
        fixture
        for fixture in fixtures
        if (fixture.team_h == team_id or fixture.team_a == team_id)
        and not fixture.finished
    ]
    team_fixtures = [
        TeamFixture(
            round=fixture.event,
            id=fixture.id,
            kickoff_time=fixture.kickoff_time,
            finished=fixture.finished,
            is_home=fixture.team_h == team_id,
            opposition=fixture.team_h
            if fixture.team_a == team_id
            else fixture.team_a,
            difficulty=fixture.team_h_difficulty
            if fixture.team_h == team_id
            else fixture.team_a_difficulty,
        )
        for fixture in unfinished_fixtures
    ]
    return team_fixtures


def predict_player_match_points(
    fixture: TeamFixture, rolling_points: float
) -> float:
    """Predict a player's points for a given fixture.

    This function will predict a player's points for a given fixture based on their rolling points.

    Parameters
    ----------
    fixture : TeamFixture
        The fixture to predict points for.
    rolling_points : float
        The player's rolling points.

    Returns
    -------
    float
        The predicted points for the player.
    """
    match fixture.difficulty:
        case 1:
            difficulty_multiplier = 1.3
        case 2:
            difficulty_multiplier = 1.15
        case 3:
            difficulty_multiplier = 1.0
        case 4:
            difficulty_multiplier = 0.85
        case 5:
            difficulty_multiplier = 0.7

    return rolling_points * difficulty_multiplier


def predict_player_future_points(
    fixtures: list[TeamFixture], player: FplPlayer
) -> list[PlayerExpectedPoints]:
    """Predict a player's points for a given game.

    Parameters
    ----------
    fixtures : list[TeamFixture]
        The fixtures to use for predictions.
    player : FplPlayer
        The player to predict points for.

    Returns
    -------
    list[PlayerExpectedPoints]
        A list of predicted points for the player.
    """
    rolling_data_filepath = DATA_FOLDER.joinpath("transformed").joinpath(
        "rolling_points.csv"
    )
    rolling_data = pl.read_csv(rolling_data_filepath)
    player_rolling_points = get_player_rolling_points(rolling_data, player)
    player_expected_points = []
    for fixture in fixtures:
        predicted_points = predict_player_match_points(
            fixture, player_rolling_points
        )
        player_expected_points.append(
            PlayerExpectedPoints(
                player=player,
                fixture=fixture,
                rolling_points=player_rolling_points,
                expected_points=predicted_points,
            )
        )
    return player_expected_points


def predict_all_players_future_points() -> list[PlayerExpectedPoints]:
    """Predict all players future points.

    Parameters
    ----------
    players : list[FplPlayer]
        The players to predict points for.

    Returns
    -------
    list[PlayerExpectedPoints]
        A list of predicted points for the players.
    """
    fpl_api = FplAPI()
    players = fpl_api.get_players()
    fixtures = fpl_api.get_fixtures()
    player_expected_points = []
    for player in players:
        player_fixtures = get_player_fixtures(fixtures, player)
        player_expected_points.extend(
            predict_player_future_points(player_fixtures, player)
        )
    return player_expected_points
