"""Which seasons actually publish which match-level column.

The two match-level tables span sources whose columns changed repeatedly.
Vaastav published a detailed stat set from 2016-17 to 2018-19, dropped it
for three seasons, then added the ``expected_*`` family in 2022-23 and
``defensive_contribution`` in 2025-26. FCI added nine columns in 2025-26
and ``corners`` in 2026-27.

A stored null therefore means one of two very different things: the stat
was published and the value was zero, or the stat did not exist that
season. These maps are the record of which. Feature code should assert
against them rather than discover the gap at training time -- a rolling
``expected_goals`` window over 2016-2026 is silently null for six
seasons otherwise.

Every season list here was measured from the source repositories.
"""

from collections.abc import Iterable

VAASTAV_SEASONS: tuple[str, ...] = (
    "2016-17",
    "2017-18",
    "2018-19",
    "2019-20",
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)

FCI_SEASONS: tuple[str, ...] = ("2024-25", "2025-26", "2026-27")

# The three eras of the detailed Vaastav stat set.
_EARLY: tuple[str, ...] = ("2016-17", "2017-18", "2018-19")
_EXPECTED: tuple[str, ...] = ("2022-23", "2023-24", "2024-25", "2025-26")
_FROM_2020_21: tuple[str, ...] = (
    "2020-21",
    "2021-22",
    "2022-23",
    "2023-24",
    "2024-25",
    "2025-26",
)

# Columns published in every Vaastav season.
_VAASTAV_ALWAYS: tuple[str, ...] = (
    "assists",
    "bonus",
    "bps",
    "clean_sheets",
    "creativity",
    "element",
    "fixture",
    "goals_conceded",
    "goals_scored",
    "gw",
    "ict_index",
    "influence",
    "kickoff_time",
    "minutes",
    "name",
    "opponent_team",
    "own_goals",
    "penalties_missed",
    "penalties_saved",
    "red_cards",
    "round",
    "saves",
    "season",
    "selected",
    "team_a_score",
    "team_h_score",
    "threat",
    "total_points",
    "transfers_balance",
    "transfers_in",
    "transfers_out",
    "value",
    "was_home",
    "yellow_cards",
)

VAASTAV_COLUMN_SEASONS: dict[str, tuple[str, ...]] = {
    **{column: VAASTAV_SEASONS for column in _VAASTAV_ALWAYS},
    "attempted_passes": _EARLY,
    "big_chances_created": _EARLY,
    "big_chances_missed": _EARLY,
    "completed_passes": _EARLY,
    "dribbles": _EARLY,
    "errors_leading_to_goal": _EARLY,
    "errors_leading_to_goal_attempt": _EARLY,
    "fouls": _EARLY,
    "key_passes": _EARLY,
    "offside": _EARLY,
    "open_play_crosses": _EARLY,
    "penalties_conceded": _EARLY,
    "tackled": _EARLY,
    "target_missed": _EARLY,
    "winning_goals": _EARLY,
    "clearances_blocks_interceptions": _EARLY + ("2025-26",),
    "recoveries": _EARLY + ("2025-26",),
    "tackles": _EARLY + ("2025-26",),
    "expected_assists": _EXPECTED,
    "expected_goal_involvements": _EXPECTED,
    "expected_goals": _EXPECTED,
    "expected_goals_conceded": _EXPECTED,
    "starts": _EXPECTED,
    "position": _FROM_2020_21,
    "team": _FROM_2020_21,
    "xP": _FROM_2020_21,
    "defensive_contribution": ("2025-26",),
    "mng_clean_sheets": ("2024-25",),
    "mng_draw": ("2024-25",),
    "mng_goals_scored": ("2024-25",),
    "mng_loss": ("2024-25",),
    "mng_underdog_draw": ("2024-25",),
    "mng_underdog_win": ("2024-25",),
    "mng_win": ("2024-25",),
}

# Columns FCI added in 2025-26.
_FCI_FROM_2025_26: tuple[str, ...] = ("2025-26", "2026-27")

_FCI_ADDED_2025_26: tuple[str, ...] = (
    "defensive_contributions",
    "dispossessed",
    "distance_covered",
    "number_of_sprints",
    "running_distance",
    "saves_inside_box",
    "sprinting_distance",
    "top_speed",
    "walking_distance",
)

# Columns present in every FCI season, plus the three we derive
# (``season``, ``gw``, ``competition``) which exist for all of them.
_FCI_ALWAYS: tuple[str, ...] = (
    "accurate_crosses",
    "accurate_crosses_percent",
    "accurate_long_balls",
    "accurate_long_balls_percent",
    "accurate_passes",
    "accurate_passes_percent",
    "aerial_duels_won",
    "aerial_duels_won_percent",
    "assists",
    "big_chances_missed",
    "blocks",
    "chances_created",
    "clearances",
    "competition",
    "dribbled_past",
    "duels_lost",
    "duels_won",
    "element",
    "final_third_passes",
    "finish_min",
    "fouls_committed",
    "gk_accurate_long_balls",
    "gk_accurate_passes",
    "goals",
    "goals_conceded",
    "goals_prevented",
    "ground_duels_won",
    "ground_duels_won_percent",
    "gw",
    "headed_clearances",
    "high_claim",
    "interceptions",
    "match_id",
    "minutes_played",
    "offsides",
    "penalties_missed",
    "penalties_scored",
    "recoveries",
    "saves",
    "season",
    "shots_on_target",
    "start_min",
    "successful_dribbles",
    "successful_dribbles_percent",
    "sweeper_actions",
    "tackles",
    "tackles_won",
    "tackles_won_percent",
    "team_goals_conceded",
    "total_shots",
    "touches",
    "touches_opposition_box",
    "was_fouled",
    "xa",
    "xg",
    "xgot",
    "xgot_faced",
)

FCI_COLUMN_SEASONS: dict[str, tuple[str, ...]] = {
    **{column: FCI_SEASONS for column in _FCI_ALWAYS},
    **{column: _FCI_FROM_2025_26 for column in _FCI_ADDED_2025_26},
    "corners": ("2026-27",),
}

# Columns FCI declares but leaves entirely null upstream. They are stored
# so they populate silently if FCI ever starts filling them, but nothing
# should model on them today.
FCI_EMPTY_COLUMNS: frozenset[str] = frozenset(
    {
        "distance_covered",
        "number_of_sprints",
        "running_distance",
        "sprinting_distance",
        "top_speed",
        "walking_distance",
    }
)


def seasons_covering(
    coverage: dict[str, tuple[str, ...]], columns: Iterable[str]
) -> tuple[str, ...]:
    """Return the seasons in which every named column is published.

    Parameters
    ----------
    coverage : dict[str, tuple[str, ...]]
        A coverage map, e.g. ``VAASTAV_COLUMN_SEASONS``.
    columns : Iterable[str]
        Columns that must all be present. An empty iterable yields every
        season the map mentions.

    Returns
    -------
    tuple[str, ...]
        The intersection of the columns' coverage, sorted.

    Raises
    ------
    KeyError
        If a column is absent from the map. An unknown column is a
        programming error and must not silently return no seasons.
    """
    names = list(columns)
    if not names:
        return tuple(sorted({s for v in coverage.values() for s in v}))
    covered = set(coverage[names[0]])
    for name in names[1:]:
        covered &= set(coverage[name])
    return tuple(sorted(covered))
