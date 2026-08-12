"""Rolling team-level attacking and defensive form.

Match-grain like ``features/match_form.py``, and offset the same way: the
window is ``ROWS BETWEEN n PRECEDING AND 1 PRECEDING``, so a team's form
never includes the match it is attached to. A second, inclusive frame is
also registered, under ``team_match_form_inclusive``: it exists solely
for the forward path, which as-of joins an unplayed fixture back to a
team's most recent match and needs that match's own figures, not the
form as of one match earlier. See :func:`window_frame` and
:func:`register_team_form`.

Two views are registered. ``team_match`` reduces the per-player FCI rows
to one row per team per fixture, carrying xG and goals both for and
against. ``team_match_form`` windows those four measures over each team's
preceding matches.

The measures are per-match means, not per-90 rates: a team plays the full
match by definition, so there is no minutes denominator to divide by --
unlike the player-level module, where a rate is the only way to compare a
starter with a substitute.

Three derivations are worth stating.

*Goals come from ``team_goals_conceded``, not from summing player goals.*
An own goal is conceded by a team but credited to no player, so a summed
``goals`` column undercounts. Measured across both FCI seasons, the
declared column is never below the opponent's summed player goals: 1434
of 1512 team-pairs agree exactly, 71 differ by one and 5 by two -- own
goals. ``goals_against`` is therefore the maximum ``team_goals_conceded``
on a team's own side, and ``goals_for`` is the opponent's
``goals_against``. The maximum is what recovers the full-match total from
a column that is really "goals conceded while this player was on the
pitch"; at least one player, normally the keeper, is on for all of it.

*``clean_sheet`` is the team's, not FPL's.* It is 1 when the team
conceded nothing, so its rolling average is a clean-sheet rate over the
window. FPL only awards a defender clean-sheet points if they also played
60 minutes; that condition belongs to the player, not to this table, and
``player_match.minutes`` is where to apply it.

*xG has no team-level column*, so ``xg_for`` is the sum over the team's
players and ``xg_against`` is the opponent's ``xg_for``. That makes it
the one measure here sensitive to a missing player row, which is what
``players`` records.

*Team names come from the fixture slug*, resolved through
``FCI_SLUG_TO_FPL``, rather than from ``player_week.team``. The slug map
holds one canonical name per club, so a club FPL renames between seasons
(``"Ipswich"`` to ``"Ipswich Town"``) does not split into two partitions
and lose its history mid-window.

One known contamination: assigning players to sides still runs through
``opta_match.is_home``, which reads ``player_week.team``. A player whose
stored club disagrees with the fixture lands on the wrong side. There is
one such row across both seasons -- a Southampton player inside an
Everton-Liverpool fixture -- carrying 0.0 xG, so it moves nothing today,
but ``players`` is exposed so an unusually sized squad is visible.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import ROLLING_WINDOW
from fantasy_football.features.naming import rolling_column_name

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# The measures windowed, in output order. ``clean_sheet`` is 0/1 per
# match, so its rolling average is a clean-sheet rate over the window.
TEAM_MEASURES: tuple[str, ...] = (
    "xg_for",
    "xg_against",
    "goals_for",
    "goals_against",
    "clean_sheet",
)

# Columns the form view adds beyond the rolling measures.
TEAM_CONTEXT_COLUMNS: tuple[str, ...] = (
    "form_matches",
    "days_since_last_match",
)

_MATCH_VIEW = "team_match"
_FORM_VIEW = "team_match_form"
_INCLUSIVE_FORM_VIEW = "team_match_form_inclusive"

# One row per team per fixture. The self-join flips each side's own
# figures into the other's "against" columns.
_TEAM_MATCH_SQL = f"""
CREATE OR REPLACE TEMP VIEW {_MATCH_VIEW} AS
WITH sides AS (
    SELECT
        o.season,
        o.gw,
        o.match_id,
        o.is_home,
        sum(o.xg)                     AS xg_for,
        max(o.team_goals_conceded)    AS goals_against,
        count(*)                      AS players
    FROM opta_match AS o
    GROUP BY o.season, o.gw, o.match_id, o.is_home
),
named AS (
    SELECT
        s.*,
        CASE WHEN s.is_home THEN home.team ELSE away.team END AS team,
        CASE WHEN s.is_home THEN away.team ELSE home.team END AS opposition
    FROM sides AS s
    LEFT JOIN fci_slug_map AS home
        ON home.slug = regexp_extract(
            s.match_id, '^\\d{{2}}-\\d{{2}}-prem-(.+)-vs-(.+)$', 1)
    LEFT JOIN fci_slug_map AS away
        ON away.slug = regexp_extract(
            s.match_id, '^\\d{{2}}-\\d{{2}}-prem-(.+)-vs-(.+)$', 2)
)
SELECT
    n.season,
    n.gw,
    n.match_id,
    n.team,
    n.opposition,
    n.is_home,
    tf.kickoff_time,
    n.xg_for,
    opp.xg_for        AS xg_against,
    opp.goals_against AS goals_for,
    n.goals_against,
    -- Null-guarded so a fixture whose concessions are unknown stays
    -- unknown rather than counting as a clean sheet kept.
    CASE
        WHEN n.goals_against IS NULL THEN NULL
        WHEN n.goals_against = 0 THEN 1
        ELSE 0
    END AS clean_sheet,
    n.players,
    opp.players       AS opposition_players
FROM named AS n
LEFT JOIN named AS opp
    ON  opp.match_id = n.match_id
    AND opp.is_home <> n.is_home
LEFT JOIN team_fixture AS tf
    ON  tf.season     = n.season
    AND tf.gw         = n.gw
    AND tf.team       = n.team
    AND tf.opposition = n.opposition
"""


def measure_column_name(measure: str, rolling_window: int) -> str:
    """Return the rolling column name for a team measure.

    Parameters
    ----------
    measure : str
        One of ``TEAM_MEASURES``.
    rolling_window : int
        Number of preceding matches in the window.

    Returns
    -------
    str
        For example ``"xg_for_rolling_5"``.
    """
    return rolling_column_name(measure, rolling_window)


def feature_columns(rolling_window: int = ROLLING_WINDOW) -> list[str]:
    """Return every column the form view adds, in view order.

    Parameters
    ----------
    rolling_window : int, optional
        Number of preceding matches in the window.

    Returns
    -------
    list[str]
        Rolling measures followed by ``TEAM_CONTEXT_COLUMNS``.
    """
    return [
        measure_column_name(measure, rolling_window)
        for measure in TEAM_MEASURES
    ] + list(TEAM_CONTEXT_COLUMNS)


def window_frame(rolling_window: int, inclusive: bool) -> str:
    """Return the ROWS frame clause for a rolling window.

    The exclusive frame ends one row before the current one, so a
    training row cannot see its own match. The inclusive frame ends on
    the current row and is used only when as-of joining the most recent
    played match onto an unplayed fixture -- there, that match has
    happened and excluding it would make the prediction one game stale.

    Parameters
    ----------
    rolling_window : int
        Number of matches the window spans.
    inclusive : bool
        Whether the current row counts towards its own window.

    Returns
    -------
    str
        A ``ROWS BETWEEN ...`` clause.
    """
    if inclusive:
        return f"ROWS BETWEEN {rolling_window - 1} PRECEDING AND CURRENT ROW"
    return f"ROWS BETWEEN {rolling_window} PRECEDING AND 1 PRECEDING"


def form_sql(
    rolling_window: int = ROLLING_WINDOW, inclusive: bool = False
) -> str:
    """Return the SELECT behind the team form view.

    Parameters
    ----------
    rolling_window : int, optional
        Number of preceding matches in the window.
    inclusive : bool, optional
        When True the current match counts towards its own window. Used
        only on the forward-scoring path; see :func:`window_frame`.

    Returns
    -------
    str
        A SELECT over ``team_match``.
    """
    averages = ",\n    ".join(
        f"avg({measure}) OVER form "
        f"AS {measure_column_name(measure, rolling_window)}"
        for measure in TEAM_MEASURES
    )
    # Under the inclusive frame, days_since_last_match is always 0,
    # because the current match is inside its own window. The forward
    # path recomputes staleness against the future fixture's kickoff and
    # ignores this column, so the exclusive view is unaffected and its
    # SQL text is unchanged.
    return f"""
SELECT
    season,
    gw,
    match_id,
    team,
    opposition,
    is_home,
    kickoff_time,
    {", ".join(TEAM_MEASURES)},
    {averages},
    count(*) OVER form AS form_matches,
    date_diff('day', max(kickoff_time) OVER form, kickoff_time)
        AS days_since_last_match
FROM {_MATCH_VIEW}
WINDOW form AS (
    PARTITION BY team
    ORDER BY kickoff_time
    {window_frame(rolling_window, inclusive)}
)
"""


def register_team_form(
    connection: "DuckDBPyConnection",
    rolling_window: int = ROLLING_WINDOW,
    inclusive: bool = False,
) -> None:
    """Create the ``team_match`` and team form views.

    Requires ``storage.lookups.register_lookups`` on the same connection.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    rolling_window : int, optional
        Number of preceding matches in the window.
    inclusive : bool, optional
        When True, register ``team_match_form_inclusive`` built on the
        inclusive frame instead of ``team_match_form``.
    """
    connection.execute(_TEAM_MATCH_SQL)
    view = _INCLUSIVE_FORM_VIEW if inclusive else _FORM_VIEW
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {view} AS "
        f"{form_sql(rolling_window, inclusive)}"
    )
    # team_match is rebuilt identically by both calls this function makes
    # per connection (exclusive, then inclusive); only warn once, on the
    # first, so a real database doesn't log the same count twice.
    if inclusive:
        return
    unordered = connection.sql(
        f"SELECT count(*) FROM {_MATCH_VIEW} WHERE kickoff_time IS NULL"
    ).fetchone()[0]
    if unordered:
        logger.warning(
            "%d team_match rows have no kickoff_time and so no position in "
            "the rolling window; their fixture is missing from "
            "team_fixture.",
            unordered,
        )


def load_team_form(
    connection: "DuckDBPyConnection",
    rolling_window: int = ROLLING_WINDOW,
) -> pl.DataFrame:
    """Return the rolling team-form frame.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection with ``register_lookups`` already run.
    rolling_window : int, optional
        Number of preceding matches in the window.

    Returns
    -------
    pl.DataFrame
        One row per team per fixture, ordered by team then kickoff.
    """
    register_team_form(connection, rolling_window)
    return connection.sql(
        f"SELECT * FROM {_FORM_VIEW} ORDER BY team, kickoff_time"
    ).pl()
