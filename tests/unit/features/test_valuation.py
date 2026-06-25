import polars as pl
import pytest

from fantasy_football.features.valuation import (
    add_positional_value_rank,
    add_team_value,
)


@pytest.fixture
def sample_value_data() -> pl.DataFrame:
    """Sample player-week data spanning two teams across two seasons.

    Returns
    -------
    pl.DataFrame
        Player-week rows with ``season``, ``gw``, ``team``, ``element``,
        ``name`` and ``value``.
    """
    return pl.DataFrame(
        {
            "season": ["2020-21", "2020-21", "2020-21", "2021-22"],
            "gw": [1, 1, 1, 1],
            "team": ["Arsenal", "Arsenal", "Chelsea", "Arsenal"],
            "element": [1, 2, 3, 1],
            "name": ["Player1", "Player2", "Player3", "Player1"],
            "value": [50, 30, 70, 60],
        }
    )


def test_add_team_value_sums_per_team_gameweek(
    sample_value_data: pl.DataFrame,
) -> None:
    """team_value is the sum of player values within each team-gameweek."""
    result = add_team_value(sample_value_data)

    by_element = {
        (row["season"], row["element"]): row["team_value"]
        for row in result.iter_rows(named=True)
    }
    # Arsenal 2020-21 GW1: 50 + 30 = 80
    assert by_element[("2020-21", 1)] == 80
    assert by_element[("2020-21", 2)] == 80
    # Chelsea 2020-21 GW1: 70 only
    assert by_element[("2020-21", 3)] == 70


def test_add_team_value_partitions_by_season(
    sample_value_data: pl.DataFrame,
) -> None:
    """Same gw/team in a different season is not pooled into the total."""
    result = add_team_value(sample_value_data)

    arsenal_2021 = result.filter(
        (pl.col("season") == "2021-22") & (pl.col("element") == 1)
    )
    # Arsenal 2021-22 GW1 has a single player worth 60 — not mixed with 2020-21.
    assert arsenal_2021.get_column("team_value").item() == 60


def test_add_team_value_computes_share(
    sample_value_data: pl.DataFrame,
) -> None:
    """value_share_of_team is the player's value over the team total."""
    result = add_team_value(sample_value_data)

    player1 = result.filter(
        (pl.col("season") == "2020-21") & (pl.col("element") == 1)
    )
    # 50 / 80
    assert player1.get_column("value_share_of_team").item() == pytest.approx(
        0.625
    )


def test_add_team_value_null_share_for_null_value() -> None:
    """A null individual value yields a null share but still counts the team."""
    data = pl.DataFrame(
        {
            "season": ["2020-21", "2020-21"],
            "gw": [1, 1],
            "team": ["Arsenal", "Arsenal"],
            "element": [1, 2],
            "name": ["Player1", "Player2"],
            "value": [50, None],
        }
    )

    result = add_team_value(data)

    # Null value is skipped in the sum, so team_value is just 50.
    assert result.get_column("team_value").to_list() == [50, 50]
    shares = result.get_column("value_share_of_team").to_list()
    assert shares[0] == pytest.approx(1.0)
    assert shares[1] is None


def test_add_positional_value_rank_ranks_within_team_position() -> None:
    """Higher value gets rank 1 within (season, gw, team, position)."""
    data = pl.DataFrame(
        {
            "season": ["2022-23"] * 4,
            "gw": [1, 1, 1, 1],
            "team": ["Arsenal", "Arsenal", "Arsenal", "Chelsea"],
            "position": ["MID", "MID", "DEF", "MID"],
            "element": [1, 2, 3, 4],
            "value": [70, 50, 40, 90],
        }
    )

    result = add_positional_value_rank(data)
    by_element = {
        row["element"]: (row["pos_value_rank"], row["players_same_pos"])
        for row in result.iter_rows(named=True)
    }

    # Arsenal MID: 70 -> rank 1, 50 -> rank 2; two players in the group.
    assert by_element[1] == (1, 2)
    assert by_element[2] == (2, 2)
    # Arsenal DEF: only player -> rank 1, group size 1.
    assert by_element[3] == (1, 1)
    # Chelsea MID: only player -> rank 1, group size 1.
    assert by_element[4] == (1, 1)


def test_add_positional_value_rank_ties_share_lower_rank() -> None:
    """Equal values share the same (minimum) rank, SQL RANK() style."""
    data = pl.DataFrame(
        {
            "season": ["2022-23"] * 3,
            "gw": [1, 1, 1],
            "team": ["Arsenal", "Arsenal", "Arsenal"],
            "position": ["MID", "MID", "MID"],
            "element": [1, 2, 3],
            "value": [50, 50, 40],
        }
    )

    result = add_positional_value_rank(data)
    by_element = {
        row["element"]: row["pos_value_rank"]
        for row in result.iter_rows(named=True)
    }

    # Two players tied at 50 both get rank 1; the 40 player gets rank 3.
    assert by_element[1] == 1
    assert by_element[2] == 1
    assert by_element[3] == 3
