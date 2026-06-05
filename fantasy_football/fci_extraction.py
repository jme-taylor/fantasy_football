"""Extract current-season FPL data from FPL Core Insights (FCI).

FCI replaces Vaastav as the data source from 2025-26 onwards. Its per-gameweek
folders hold mostly season-to-date snapshots, so this module reconstructs
Vaastav-shaped ``merged_gw.csv`` rows (one row per player per gameweek) from
them. The pure ``build_merged_gw`` adapter does the reshaping; the IO and
orchestration live in ``FciExtractor`` (added separately).
"""

import logging

import polars as pl

logger = logging.getLogger(__name__)

# FCI position labels -> Vaastav/FPL short codes.
FCI_POSITION_TO_VAASTAV: dict[str, str] = {
    "Goalkeeper": "GK",
    "Defender": "DEF",
    "Midfielder": "MID",
    "Forward": "FWD",
}

MERGED_GW_COLUMNS: list[str] = [
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
]


def build_merged_gw(
    snapshots: pl.DataFrame,
    matchstats: pl.DataFrame,
    players: pl.DataFrame,
    team_code_to_name: dict[int, str],
) -> pl.DataFrame:
    """Reconstruct Vaastav-shaped per-gameweek rows from FCI frames.

    Parameters
    ----------
    snapshots : pl.DataFrame
        Concatenated ``player_gameweek_stats`` across gameweeks, with an added
        ``gw`` column. Must contain ``gw, id, first_name, second_name,
        now_cost, event_points, bonus`` (``bonus`` cumulative for the season).
    matchstats : pl.DataFrame
        Concatenated ``playermatchstats`` with a ``gw`` column. Must contain
        ``gw, player_id, minutes_played``.
    players : pl.DataFrame
        Season ``players`` file. Must contain ``player_id, position,
        team_code``.
    team_code_to_name : dict[int, str]
        Maps FCI ``team_code`` (== FPL team ``code``) to official team name.

    Returns
    -------
    pl.DataFrame
        One row per (player, gameweek) with columns ``MERGED_GW_COLUMNS``.
    """
    # Per-GW minutes: sum across matches so double gameweeks accumulate.
    minutes = matchstats.group_by(["gw", "player_id"]).agg(
        pl.col("minutes_played").sum().alias("minutes")
    )

    # Event-level bonus: difference of cumulative season bonus per player; the
    # first gameweek keeps its cumulative value.
    snap = snapshots.sort(["id", "gw"]).with_columns(
        (pl.col("bonus") - pl.col("bonus").shift(1).over("id"))
        .fill_null(pl.col("bonus"))
        .alias("event_bonus")
    )

    # Map team_code -> name via a small mapping frame join (this polars version
    # lacks ``replace_strict``, so we join rather than replace-with-default).
    team_map = pl.DataFrame(
        {
            "team_code": list(team_code_to_name.keys()),
            "team": list(team_code_to_name.values()),
        }
    )

    merged = (
        snap.join(
            players.select("player_id", "position", "team_code"),
            left_on="id",
            right_on="player_id",
            how="left",
            coalesce=True,
        )
        .join(team_map, on="team_code", how="left", coalesce=True)
        .join(
            minutes,
            left_on=["gw", "id"],
            right_on=["gw", "player_id"],
            how="left",
            coalesce=True,
        )
        .with_columns(
            (pl.col("first_name") + " " + pl.col("second_name")).alias("name"),
            pl.col("position").replace(FCI_POSITION_TO_VAASTAV),
            pl.col("id").alias("element"),
            pl.col("minutes").fill_null(0).cast(pl.Int64),
            pl.col("gw").alias("round"),
            pl.col("event_points").alias("total_points"),
            pl.col("gw").alias("GW"),
            (pl.col("now_cost") * 10).round(0).cast(pl.Int64).alias("value"),
            pl.col("event_bonus").cast(pl.Int64).alias("bonus"),
        )
    )
    return merged.select(MERGED_GW_COLUMNS)
