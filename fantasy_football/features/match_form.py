"""Rolling per-90 form over a player's preceding matches.

These are match-grain features: they window over ``player_match`` legs, so
the two halves of a double gameweek are separate observations rather than
one summed row. That is the whole reason for the ``opta_match`` bridge in
``storage/lookups.py``, which this module builds on.

Three choices are worth stating up front.

*Minutes come from ``player_match``, never from ``player_match_opta``.*
The two sources disagree by a minute on roughly 13% of rows (FCI rounds
differently), and FPL's minutes are what scoring is settled on, so they
are the source of truth for both the appearance filter and every per-90
denominator.

*The window spans appearances, not fixtures.* A benched player's window
would otherwise fill with 0-minute squad entries and his form would go
null while three real matches sat just outside it. Instead the window
always covers his last ``rolling_window`` appearances, however long ago,
and ``days_since_last_appearance`` reports how stale that is -- the same
carry-the-value-and-flag-its-age shape ``features/history.py`` uses for
``prev_season_*`` and ``seasons_since_last_pl``.

*Each stat gets its own denominator.* A match contributes minutes to a
stat's per-90 rate only when that stat is non-null in it. Nulls here mean
"not published", not zero: ``defensive_contributions`` is null for all of
2024-25, and a handful of matches have no FCI row at all because FCI
filed a rearranged fixture under a different gameweek than FPL. A shared
denominator would divide a partial numerator by full minutes and quietly
report a rate that is too low; a per-stat denominator returns null
instead, which is the honest answer.

The window frame is ``ROWS BETWEEN n PRECEDING AND 1 PRECEDING``, so the
current match is excluded by construction -- there is no shift to forget.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import ROLLING_WINDOW
from fantasy_football.features.transformation import (
    _FALLBACK_IDENTITY_PREFIX,
    rolling_column_name,
)
from fantasy_football.storage.coverage import (
    FCI_COLUMN_SEASONS,
    FCI_EMPTY_COLUMNS,
    seasons_covering,
)
from fantasy_football.storage.tables import (
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# The FCI stats windowed into per-90 rates. Defender-weighted: the first
# two are the attacking-return signal, the rest are the defensive
# workload a clean-sheet or defensive-contribution model needs.
PER90_STATS: tuple[str, ...] = (
    "xg",
    "xa",
    "tackles",
    "interceptions",
    "clearances",
    "blocks",
    "recoveries",
    "defensive_contributions",
)

# Stats windowed the same way but sourced from ``player_match_fpl``.
# FCI publishes no cards at all, and Vaastav's go back to 2016-17 -- six
# seasons further than any FCI column.
FPL_PER90_STATS: tuple[str, ...] = (
    "yellow_cards",
    "red_cards",
)

# Stats also accumulated across the season to date. Cards are the case
# that needs it: a per-90 rate says how freely a player is booked, but
# suspensions are triggered by a running count, and that count resets
# each season -- so the cumulative window is season-scoped where the
# rolling one is not.
CUMULATIVE_STATS: tuple[str, ...] = (
    "yellow_cards",
    "red_cards",
)

# Columns the view adds beyond the per-90 rates.
FORM_CONTEXT_COLUMNS: tuple[str, ...] = (
    "form_matches",
    "form_minutes",
    "days_since_last_appearance",
)

# Columns a per-90 rate is meaningless for. Percentages and speeds are
# already rates, and the minute markers are instants; dividing any of
# them by minutes produces a number that looks fine and means nothing.
# ``minutes_played`` is excluded because FPL's minutes are the source of
# truth for the denominator -- see the module docstring.
_NON_RATEABLE: frozenset[str] = frozenset(
    column
    for column in PLAYER_MATCH_OPTA.schema
    if column.endswith("_percent")
) | frozenset(
    {
        # Keys and metadata on either source table.
        "season",
        "gw",
        "element",
        "match_id",
        "competition",
        "fixture",
        "opponent_team",
        "name",
        "round",
        "kickoff_time",
        "was_home",
        # Already a rate, an instant, or a price rather than a count.
        "minutes_played",
        "minutes",
        "start_min",
        "finish_min",
        "top_speed",
        "value",
        "selected",
        "transfers_balance",
        "transfers_in",
        "transfers_out",
    }
)

_VIEW_NAME = "player_match_form"


def validate_stats(
    stats: "tuple[str, ...] | list[str]",
    source: str = "opta",
) -> None:
    """Raise if any stat cannot produce a meaningful per-90 rate.

    Adding a column to ``PER90_STATS`` is a one-line change, and three of
    the ways it can go wrong are silent: a misspelling only surfaces as a
    DuckDB binder error at query time, a ``*_percent`` column yields a
    plausible-looking number that means nothing, and a column FCI declares
    but never fills yields nulls that look like missing data rather than a
    modelling mistake. This turns all three into an error naming the
    remedy, as ``storage/coverage.py`` asks feature code to do.

    Parameters
    ----------
    stats : tuple[str, ...] | list[str]
        Candidate column names.
    source : str, optional
        ``"opta"`` to check against ``player_match_opta`` (the default),
        ``"fpl"`` for ``player_match_fpl``. The two tables publish
        different stats -- only Vaastav has cards, only FCI has xG -- so
        a name valid for one is usually a mistake for the other.

    Raises
    ------
    ValueError
        If a stat is unknown, not rateable, or never populated.
    """
    table = PLAYER_MATCH_FPL if source == "fpl" else PLAYER_MATCH_OPTA
    seen: set[str] = set()
    repeated = sorted({s for s in stats if s in seen or seen.add(s)})
    if repeated:
        raise ValueError(
            f"Listed more than once: {', '.join(repeated)}. Each would "
            "emit a duplicate column of the same name."
        )
    unknown = [s for s in stats if s not in table.schema]
    if unknown:
        raise ValueError(
            f"Not {table.name} columns: {', '.join(sorted(unknown))}. "
            "Check the spelling against storage/tables.py."
        )
    not_rateable = [s for s in stats if s in _NON_RATEABLE]
    if not_rateable:
        raise ValueError(
            f"Per-90 is meaningless for: {', '.join(sorted(not_rateable))}. "
            "Percentages and speeds are already rates; window them with a "
            "minutes-weighted mean instead of this module."
        )
    empty = [s for s in stats if s in FCI_EMPTY_COLUMNS]
    if empty:
        raise ValueError(
            f"FCI declares but never populates: {', '.join(sorted(empty))}. "
            "Every rate would be null. Remove them, or drop them from "
            "FCI_EMPTY_COLUMNS once FCI starts filling them."
        )


def covered_seasons(
    stats: "tuple[str, ...] | list[str] | None" = None,
) -> tuple[str, ...]:
    """Return the seasons in which every stat is actually published.

    The rolling window spans seasons, so a stat added in 2025-26 is null
    for every window reaching back into 2024-25. This reports the usable
    range up front rather than leaving it to be discovered at training
    time.

    Parameters
    ----------
    stats : tuple[str, ...] | list[str] | None, optional
        Defaults to ``PER90_STATS``.

    Returns
    -------
    tuple[str, ...]
        Sorted seasons covering every named stat.
    """
    return seasons_covering(FCI_COLUMN_SEASONS, stats or PER90_STATS)


def per90_column_name(stat: str, rolling_window: int) -> str:
    """Return the output column name for a stat's rolling per-90 rate.

    Reuses ``rolling_column_name`` so these line up with the existing
    ``*_rolling_5`` feature names.

    Parameters
    ----------
    stat : str
        A column of ``player_match_opta``, e.g. ``"xg"``.
    rolling_window : int
        Number of preceding appearances in the window.

    Returns
    -------
    str
        For example ``"xg_per90_rolling_5"``.
    """
    return rolling_column_name(f"{stat}_per90", rolling_window)


def cumulative_column_name(stat: str) -> str:
    """Return the output column name for a stat's season-to-date total.

    Parameters
    ----------
    stat : str
        A column of ``player_match_fpl``, e.g. ``"yellow_cards"``.

    Returns
    -------
    str
        For example ``"yellow_cards_season_to_date"``. Named for the
        window rather than "cumulative", because the count resets each
        season and a bare "cumulative" would read as career-long.
    """
    return f"{stat}_season_to_date"


def feature_columns(rolling_window: int = ROLLING_WINDOW) -> list[str]:
    """Return every feature column the view adds, in view order.

    Parameters
    ----------
    rolling_window : int, optional
        Number of preceding appearances in the window.

    Returns
    -------
    list[str]
        Per-90 rate columns followed by ``FORM_CONTEXT_COLUMNS``.
    """
    return (
        [
            per90_column_name(stat, rolling_window)
            for stat in (*PER90_STATS, *FPL_PER90_STATS)
        ]
        + [cumulative_column_name(stat) for stat in CUMULATIVE_STATS]
        + list(FORM_CONTEXT_COLUMNS)
    )


def _identity_sql() -> str:
    """Return the SQL mirroring ``add_rolling_identity_column``.

    Partitioning on ``player_code`` alone would pool every player with a
    null one into a single window -- SQL groups nulls together exactly as
    polars does -- averaging strangers' form into each other. The fallback
    is scoped to ``(season, element)`` because ``element`` is unique only
    within a season, and prefixed so it can never collide with a real
    ``player_code``, which renders as bare digits.
    """
    return (
        "CASE WHEN s.player_code IS NOT NULL "
        "THEN CAST(s.player_code AS VARCHAR) "
        f"ELSE '{_FALLBACK_IDENTITY_PREFIX}' || m.season || '_' "
        "|| CAST(m.element AS VARCHAR) END"
    )


def form_sql(rolling_window: int = ROLLING_WINDOW) -> str:
    """Return the SELECT statement behind the ``player_match_form`` view.

    Parameters
    ----------
    rolling_window : int, optional
        Number of preceding appearances in the window.

    Returns
    -------
    str
        A SELECT over ``player_match``, ``player_season`` and
        ``opta_match``.
    """
    validate_stats(PER90_STATS)
    validate_stats(FPL_PER90_STATS, source="fpl")
    validate_stats(CUMULATIVE_STATS, source="fpl")
    rates = ",\n        ".join(
        f"90.0 * sum(a.{stat}) OVER form "
        f"/ nullif(sum(CASE WHEN a.{stat} IS NOT NULL THEN a.minutes END) "
        f"OVER form, 0) AS {per90_column_name(stat, rolling_window)}"
        for stat in (*PER90_STATS, *FPL_PER90_STATS)
    )
    # Season-scoped and offset by one, so a booking counts towards every
    # later match in the season but never towards its own.
    # Coalesced to 0: an empty window is the season's first appearance,
    # where the running total genuinely is zero. That differs from the
    # rolling rates, where an empty window means "no evidence" and has to
    # stay null.
    totals = ",\n    ".join(
        f"coalesce(sum(a.{stat}) OVER season_to_date, 0) "
        f"AS {cumulative_column_name(stat)}"
        for stat in CUMULATIVE_STATS
    )
    stat_columns = ",\n            ".join(
        [f"o.{stat}" for stat in PER90_STATS]
        + [
            f"f.{stat}"
            for stat in dict.fromkeys((*FPL_PER90_STATS, *CUMULATIVE_STATS))
        ]
    )
    return f"""
WITH appearances AS (
    SELECT
        m.season,
        m.gw,
        m.element,
        m.opponent,
        m.is_home,
        m.kickoff_time,
        m.minutes,
        m.total_points,
        {_identity_sql()} AS rolling_identity,
        {stat_columns}
    FROM player_match AS m
    LEFT JOIN player_season AS s
        ON s.season = m.season AND s.element = m.element
    LEFT JOIN {_VIEW_NAME}_opta AS o
        ON  o.season   = m.season
        AND o.gw       = m.gw
        AND o.element  = m.element
        AND o.opponent = m.opponent
    -- opponent_team is the same FPL team id as player_match.opponent, so
    -- this needs no bridge. It does need to be in the join: without it
    -- both legs of a double gameweek match both rows.
    LEFT JOIN player_match_fpl AS f
        ON  f.season        = m.season
        AND f.gw            = m.gw
        AND f.element       = m.element
        AND f.opponent_team = m.opponent
    WHERE m.minutes > 0
)
SELECT
    a.season,
    a.gw,
    a.element,
    a.opponent,
    a.is_home,
    a.kickoff_time,
    a.minutes,
    a.total_points,
    {rates},
    {totals},
    count(*) OVER form AS form_matches,
    sum(a.minutes) OVER form AS form_minutes,
    date_diff('day', max(a.kickoff_time) OVER form, a.kickoff_time)
        AS days_since_last_appearance
FROM appearances AS a
WINDOW
    form AS (
        PARTITION BY a.rolling_identity
        ORDER BY a.kickoff_time
        ROWS BETWEEN {rolling_window} PRECEDING AND 1 PRECEDING
    ),
    season_to_date AS (
        PARTITION BY a.rolling_identity, a.season
        ORDER BY a.kickoff_time
        ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING
    )
"""


def register_match_form(
    connection: "DuckDBPyConnection",
    rolling_window: int = ROLLING_WINDOW,
) -> None:
    """Create the ``player_match_form`` view.

    Requires ``storage.lookups.register_lookups`` to have run on the same
    connection; the view reads ``opta_match``.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    rolling_window : int, optional
        Number of preceding appearances in the window.
    """
    # opta_match is aliased so this module reads one name, whether the
    # bridge view is renamed later or swapped for a materialised table.
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {_VIEW_NAME}_opta AS "
        "SELECT * FROM opta_match"
    )
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {_VIEW_NAME} AS "
        f"{form_sql(rolling_window)}"
    )


def load_match_form(
    connection: "DuckDBPyConnection",
    rolling_window: int = ROLLING_WINDOW,
) -> pl.DataFrame:
    """Return the rolling per-90 form frame.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection with ``register_lookups`` already run.
    rolling_window : int, optional
        Number of preceding appearances in the window.

    Returns
    -------
    pl.DataFrame
        One row per appearance, ordered by player then kickoff.
    """
    register_match_form(connection, rolling_window)
    return connection.sql(
        f"SELECT * FROM {_VIEW_NAME} ORDER BY element, kickoff_time"
    ).pl()
