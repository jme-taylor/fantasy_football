"""Regression: transferred players keep their real club per gameweek.

Antoine Semenyo and Marc Guehi both transfer to Man City mid-season in this
synthetic fixture. Before the fix, ``team`` came from a static final club and
read Man City for every gameweek. The per-gameweek fplcache team_code must show
their pre-transfer clubs early and Man City only after the transfer.
"""

import polars as pl

from fantasy_football.extraction.fci import build_merged_gw

# Semenyo (element 82) and Guehi (element 200) across GW1 and GW2.
SNAPSHOTS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "id": [82, 200, 82, 200],
        "first_name": ["Antoine", "Marc", "Antoine", "Marc"],
        "second_name": ["Semenyo", "Guehi", "Semenyo", "Guehi"],
        "now_cost": [7.0, 5.0, 7.1, 5.1],
        "event_points": [6, 2, 8, 5],
        "bonus": [1, 0, 1, 1],
    },
    schema_overrides={"gw": pl.Int32},
)
MATCHSTATS = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "player_id": [82, 200, 82, 200],
        "minutes_played": [90, 90, 90, 90],
    },
    schema_overrides={"gw": pl.Int32},
)
# Static club is the FINAL club (Man City, 43) for both -- the bug source.
PLAYERS = pl.DataFrame(
    {
        "player_id": [82, 200],
        "position": ["Midfielder", "Defender"],
        "team_code": [43, 43],
    }
)
# Per-gameweek truth: Bournemouth (91) / Crystal Palace (31) in GW1; City GW2.
PLAYER_GW_TEAM = pl.DataFrame(
    {
        "gw": [1, 1, 2, 2],
        "element": [82, 200, 82, 200],
        "team_code": [91, 31, 43, 43],
    },
    schema={"gw": pl.Int64, "element": pl.Int64, "team_code": pl.Int64},
)
TEAM_CODE_TO_NAME = {
    43: "Man City",
    91: "Bournemouth",
    31: "Crystal Palace",
}


def test_transferred_players_keep_real_club_per_gameweek() -> None:
    """Semenyo/Guehi show pre-transfer clubs in GW1, Man City in GW2."""
    result = build_merged_gw(
        SNAPSHOTS, MATCHSTATS, PLAYERS, PLAYER_GW_TEAM, TEAM_CODE_TO_NAME
    ).sort(["element", "GW"])

    teams = {
        (row["name"], row["GW"]): row["team"]
        for row in result.iter_rows(named=True)
    }
    assert teams[("Antoine Semenyo", 1)] == "Bournemouth"
    assert teams[("Antoine Semenyo", 2)] == "Man City"
    assert teams[("Marc Guehi", 1)] == "Crystal Palace"
    assert teams[("Marc Guehi", 2)] == "Man City"

    # Guard against the old bug: not uniformly Man City.
    assert result.filter(pl.col("team") == "Man City").height == 2
