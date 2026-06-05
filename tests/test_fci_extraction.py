import polars as pl

from fantasy_football.fci_extraction import build_merged_gw

# Two players across two gameweeks; player 2 has a double gameweek in GW2.
SNAPSHOTS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "id": [1, 2, 1, 2],
        "first_name": ["David", "Erling", "David", "Erling"],
        "second_name": ["Raya", "Haaland", "Raya", "Haaland"],
        "now_cost": [6.0, 14.0, 6.1, 14.2],
        "event_points": [6, 9, 2, 13],
        "bonus": [1, 3, 1, 7],  # cumulative season bonus
    }
)
MATCHSTATS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2, 2],
        "player_id": [1, 2, 1, 2, 2],  # player 2 plays twice in GW2
        "match_id": ["m1", "m1", "m2", "m2", "m3"],
        "minutes_played": [90, 90, 90, 80, 30],
    }
)
PLAYERS = pl.DataFrame(
    {
        "player_id": [1, 2],
        "position": ["Goalkeeper", "Forward"],
        "team_code": [3, 43],
    }
)
TEAM_CODE_TO_NAME = {3: "Arsenal", 43: "Man City"}


def test_build_merged_gw_columns_and_shape() -> None:
    """Adapter returns exactly the merged_gw columns, one row per player-gw."""
    result = build_merged_gw(SNAPSHOTS, MATCHSTATS, PLAYERS, TEAM_CODE_TO_NAME)
    assert set(result.columns) == {
        "name",
        "position",
        "team",
        "bonus",
        "element",
        "minutes",
        "round",
        "total_points",
        "GW",
        "value",
    }
    assert result.height == 4


def test_build_merged_gw_maps_core_fields() -> None:
    """Core fields map from the FCI snapshot for a known player-gw."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, TEAM_CODE_TO_NAME
    ).sort(["GW", "element"])
    raya_gw1 = result.filter(
        (pl.col("element") == 1) & (pl.col("GW") == 1)
    ).row(0, named=True)
    assert raya_gw1["name"] == "David Raya"
    assert raya_gw1["position"] == "GK"
    assert raya_gw1["team"] == "Arsenal"
    assert raya_gw1["total_points"] == 6
    assert raya_gw1["value"] == 60  # 6.0 * 10
    assert raya_gw1["round"] == 1


def test_build_merged_gw_sums_double_gameweek_minutes() -> None:
    """Minutes are summed across a player's matches within one gameweek."""
    result = build_merged_gw(SNAPSHOTS, MATCHSTATS, PLAYERS, TEAM_CODE_TO_NAME)
    haaland_gw2 = result.filter(
        (pl.col("element") == 2) & (pl.col("GW") == 2)
    ).row(0, named=True)
    assert haaland_gw2["minutes"] == 110  # 80 + 30


def test_build_merged_gw_bonus_is_event_level_diff() -> None:
    """Event bonus is the cumulative-snapshot difference; first gw kept as-is."""
    result = build_merged_gw(SNAPSHOTS, MATCHSTATS, PLAYERS, TEAM_CODE_TO_NAME)
    # Player 2 cumulative bonus 3 -> 7, so GW2 event bonus = 4.
    haaland_gw2 = result.filter(
        (pl.col("element") == 2) & (pl.col("GW") == 2)
    ).row(0, named=True)
    assert haaland_gw2["bonus"] == 4
    # First GW uses the cumulative value as-is.
    haaland_gw1 = result.filter(
        (pl.col("element") == 2) & (pl.col("GW") == 1)
    ).row(0, named=True)
    assert haaland_gw1["bonus"] == 3


def test_build_merged_gw_fills_missing_minutes_with_zero() -> None:
    """Players with no match rows in a gameweek get zero minutes."""
    snaps = SNAPSHOTS.clone()
    # No matchstats row for player 1 in GW2 -> minutes should be 0.
    matchstats = MATCHSTATS.filter(
        ~((pl.col("gw") == 2) & (pl.col("player_id") == 1))
    )
    result = build_merged_gw(snaps, matchstats, PLAYERS, TEAM_CODE_TO_NAME)
    raya_gw2 = result.filter(
        (pl.col("element") == 1) & (pl.col("GW") == 2)
    ).row(0, named=True)
    assert raya_gw2["minutes"] == 0
