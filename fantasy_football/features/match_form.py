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

*Form is attached by time, not by match.* The rates are computed over
appearances, then as-of joined onto every ``player_match`` row -- played
or not -- from the player's last appearance strictly before that kickoff.
Keying the rates to the match they were computed in would leave a
0-minute leg with no form row at all, and null form would then encode
"he did not play", which is the outcome leaking into the inputs. The
as-of join reproduces the old values exactly for played rows: the
inclusive window ending on the previous appearance spans the same
matches the exclusive window ending one row before the current match
did.

``days_since_last_appearance`` is what stops the carry being a lie -- it
is measured against the row's own kickoff, so stale form is visibly
stale.

A second view, ``player_match_form_inclusive``, keeps the old
appearance-keyed shape on an inclusive frame. It exists solely for the
forward path, which as-of joins an unplayed *future* fixture back to a
player's most recent appearance and needs that appearance's own figures.
See ``team_form.window_frame`` and :func:`register_match_form`.
"""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import (
    DEFCON_THRESHOLD_BY_POSITION,
    ROLLING_WINDOW,
)
from fantasy_football.features.naming import (
    FALLBACK_IDENTITY_PREFIX,
    rolling_column_name,
)
from fantasy_football.features.team_form import (
    penalty_exposure_sql,
    register_team_penalty_form,
    window_frame,
)
from fantasy_football.storage.coverage import (
    FCI_COLUMN_SEASONS,
    FCI_EMPTY_COLUMNS,
    seasons_covering,
)
from fantasy_football.storage.tables import (
    PLAYER_MATCH,
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
# ``goals_scored`` is here rather than in ``PER90_STATS`` for the same
# reason the goals target is: FCI's match stats span every competition,
# so its goals column would carry a cup goal into a league gameweek.
FPL_PER90_STATS: tuple[str, ...] = (
    "goals_scored",
    "assists",
)

# Cards, sourced from the ``player_match`` spine rather than from either
# provider table. FCI has never published cards and Vaastav's per-fixture
# files stop at 2025-26, so a provider-sourced card column is null for
# the live season -- and the spine is the one relation filled for every
# season, from Vaastav historically and from the FPL API now.
MATCH_PER90_STATS: tuple[str, ...] = (
    "yellow_cards",
    "red_cards",
)

# Creation stats, read only by the assists head. Held apart from
# ``PER90_STATS`` for the reason the keeper stats are: the default
# ``covered_seasons`` window is computed over that list and drives three
# other models' fold test seasons, so a creation stat added there could
# shrink their validation for a measure none of them reads.
CREATION_PER90_STATS: tuple[str, ...] = (
    "chances_created",
    "accurate_crosses",
)

# Discipline stats, read only by the yellow-cards head. Held apart from
# ``PER90_STATS`` for the same reason the creation stats are.
DISCIPLINE_PER90_STATS: tuple[str, ...] = ("fouls_committed",)

# GK specific stats taken from Vaastav rather than FCI.
GK_FPL_PER90_STATS: tuple[str, ...] = (
    "saves",
    "penalties_saved",
    "goals_conceded",
)

# The counters FPL adds up for a defender's defensive contribution.
# Recoveries are deliberately absent: they count towards the midfield and
# forward threshold, not the defender one.
DEFCON_COMPONENT_STATS: tuple[str, ...] = (
    "clearances",
    "blocks",
    "interceptions",
    "tackles",
)

# The midfield and forward counters: the defender four, plus recoveries.
DEFCON_MID_FWD_COMPONENT_STATS: tuple[str, ...] = (
    *DEFCON_COMPONENT_STATS,
    "recoveries",
)

# Stats also accumulated across the season to date. Cards are the case
# that needs it: a per-90 rate says how freely a player is booked, but
# suspensions are triggered by a running count, and that count resets
# each season -- so the cumulative window is season-scoped where the
# rolling one is not. Sourced from ``player_match``, as
# :data:`MATCH_PER90_STATS` is.
CUMULATIVE_STATS: tuple[str, ...] = (
    "yellow_cards",
    "red_cards",
)

# Every stat the view rates, by source. The view emits one column per
# entry, so these are what ``form_sql`` and ``feature_columns`` iterate;
# the lists above are what each model picks from.
OPTA_RATE_STATS: tuple[str, ...] = (
    *PER90_STATS,
    *CREATION_PER90_STATS,
    *DISCIPLINE_PER90_STATS,
)
FPL_RATE_STATS: tuple[str, ...] = (*FPL_PER90_STATS, *GK_FPL_PER90_STATS)
MATCH_RATE_STATS: tuple[str, ...] = MATCH_PER90_STATS

#: Every rated stat, in the order the view emits them.
RATE_STATS: tuple[str, ...] = (
    *OPTA_RATE_STATS,
    *FPL_RATE_STATS,
    *MATCH_RATE_STATS,
)


@dataclass(frozen=True, slots=True)
class DefconVariant:
    """One position group's defensive-contribution count and threshold.

    FPL pays the same rule off two counts: defenders on CBIT at 10,
    midfielders and forwards on CBIRT at 12. Each gets a rate, a hit
    rate and a spread -- a rate alone cannot describe a threshold, since
    two players on the same mean clear it at different frequencies
    depending on their spread.
    """

    #: Prefix of the emitted columns, and the count's own alias.
    count: str
    #: The ``player_match_opta`` counters summed into the count.
    stats: tuple[str, ...]
    #: What the count must reach to be paid.
    threshold: int
    #: How the threshold is spelled in the hit-rate column name.
    threshold_label: str

    @property
    def form_stats(self) -> tuple[str, ...]:
        """Return the three windowed stat names, before the suffix."""
        return (
            f"{self.count}_per90",
            f"{self.count}_{self.threshold_label}_rate",
            f"{self.count}_std",
        )

    def form_columns(
        self, rolling_window: int = ROLLING_WINDOW
    ) -> tuple[str, ...]:
        """Return the emitted column names for a window.

        Named from the window they are computed over, like every other
        rate, so changing ``ROLLING_WINDOW`` cannot leave a column
        claiming five appearances while covering another number.
        """
        return tuple(
            rolling_column_name(stat, rolling_window)
            for stat in self.form_stats
        )


CBIT_VARIANT = DefconVariant(
    count="cbit",
    stats=DEFCON_COMPONENT_STATS,
    threshold=DEFCON_THRESHOLD_BY_POSITION["DEF"],
    threshold_label="ten_plus",
)

CBIRT_VARIANT = DefconVariant(
    count="cbirt",
    stats=DEFCON_MID_FWD_COMPONENT_STATS,
    threshold=DEFCON_THRESHOLD_BY_POSITION["MID"],
    threshold_label="twelve_plus",
)

#: Every variant the view emits a trio for.
DEFCON_VARIANTS: tuple[DefconVariant, ...] = (CBIT_VARIANT, CBIRT_VARIANT)


# The player's own penalty counters, summed into one attempt count.
PENALTY_COMPONENT_STATS: tuple[str, ...] = (
    "penalties_scored",
    "penalties_missed",
)

#: Expected penalty attempts per 90, built in ``features/team_form.py``.
PENALTY_EXPOSURE_COLUMN = "expected_pen_attempts_per_90"

#: 1.0 when the window carried no xG-bearing appearance. Read the
#: comment on :data:`ANCHOR_STAT` for what that does and does not mean.
NO_FORM_COLUMN = "has_no_form"

# The stat whose absence stands in for "no evidence". FCI publishes xG
# for every appearance it files, so a null rate means the window held no
# such appearance -- a debut, or a window sitting entirely before FCI
# coverage begins in 2024-25. The second case is why this reads as "no
# xG behind him" rather than "no form at all": an established player
# early in 2024-25 trips it despite a full window of appearances, which
# is honest about the feature set even though it is not a debut.
ANCHOR_STAT = "xg"


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
_INCLUSIVE_VIEW_NAME = "player_match_form_inclusive"

_VALIDATION_SOURCES = {
    "opta": PLAYER_MATCH_OPTA,
    "fpl": PLAYER_MATCH_FPL,
    "player_match": PLAYER_MATCH,
}


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
        ``"fpl"`` for ``player_match_fpl``, ``"player_match"`` for the
        spine. The tables publish different stats -- only FCI has xG,
        only the spine carries cards for every season -- so a name valid
        for one is usually a mistake for the other.

    Raises
    ------
    ValueError
        If a stat is unknown, not rateable, or never populated.
    """
    table = _VALIDATION_SOURCES.get(source, PLAYER_MATCH_OPTA)
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
        [per90_column_name(stat, rolling_window) for stat in RATE_STATS]
        + [cumulative_column_name(stat) for stat in CUMULATIVE_STATS]
        + [
            name
            for variant in DEFCON_VARIANTS
            for name in variant.form_columns(rolling_window)
        ]
        + [PENALTY_EXPOSURE_COLUMN, NO_FORM_COLUMN]
        + list(FORM_CONTEXT_COLUMNS)
    )


def rolling_identity_sql(identity: str = "s", row: str = "m") -> str:
    """Return the SQL building a collision-safe rolling identity.

    Partitioning on ``player_code`` alone would pool every player with a
    null one into a single window -- SQL groups nulls together exactly as
    polars does -- averaging strangers' form into each other. The fallback
    is scoped to ``(season, element)`` because ``element`` is unique only
    within a season, and prefixed so it can never collide with a real
    ``player_code``, which renders as bare digits.

    This is public because the forward-scoring path has to resolve the
    same key for a player who has not appeared yet, and resolving it a
    second way would let serve-time identity drift from the identity the
    view windows on.

    Parameters
    ----------
    identity : str, optional
        Alias of the relation supplying ``player_code`` -- usually
        ``player_season``.
    row : str, optional
        Alias of the relation supplying ``season`` and ``element`` for
        the row being keyed. May be the same alias as ``identity``.

    Returns
    -------
    str
        A CASE expression yielding a VARCHAR identity.
    """
    return (
        f"CASE WHEN {identity}.player_code IS NOT NULL "
        f"THEN CAST({identity}.player_code AS VARCHAR) "
        f"ELSE '{FALLBACK_IDENTITY_PREFIX}' || {row}.season || '_' "
        f"|| CAST({row}.element AS VARCHAR) END"
    )


def opta_count_sql(stats: tuple[str, ...], alias: str) -> str:
    """Return the SQL summing FCI counters into one count.

    Null when FCI published none of them, rather than zero: a player
    with no data did not make no clearances, we simply do not know. Zero
    here would drag the rolling rate down and read as a quiet player.

    ``alias`` is required: the feature view and the modelling frames
    join the same table under different names, and a default would bind
    silently to whichever module declared it.
    """
    absent = " AND ".join(f"{alias}.{stat} IS NULL" for stat in stats)
    totalled = " + ".join(f"coalesce({alias}.{stat}, 0)" for stat in stats)
    return f"CASE WHEN {absent} THEN NULL ELSE {totalled} END"


def _defcon_windowed_sql(variant: DefconVariant, rolling_window: int) -> str:
    """Return one variant's three windowed columns, as SELECT items."""
    rate, hit_rate, spread = variant.form_columns(rolling_window)
    count = variant.count
    return f"""90.0 * sum(a.{count}) OVER form
        / nullif(sum(CASE WHEN a.{count} IS NOT NULL THEN a.minutes END)
                 OVER form, 0) AS {rate},
    avg(CASE WHEN a.{count} IS NULL THEN NULL
             WHEN a.{count} >= {variant.threshold} THEN 1.0
             ELSE 0.0 END) OVER form AS {hit_rate},
    stddev_samp(a.{count}) OVER form AS {spread}"""


def form_sql(
    rolling_window: int = ROLLING_WINDOW, inclusive: bool = False
) -> str:
    """Return the SELECT statement behind the ``player_match_form`` view.

    Parameters
    ----------
    rolling_window : int, optional
        Number of preceding appearances in the window.
    inclusive : bool, optional
        When True the current appearance counts towards its own window.
        Used only on the forward-scoring path; see
        :func:`fantasy_football.features.team_form.window_frame`.

    Returns
    -------
    str
        A SELECT over ``player_match``, ``player_season`` and
        ``opta_match``.
    """
    validate_stats(OPTA_RATE_STATS)
    validate_stats(FPL_RATE_STATS, source="fpl")
    validate_stats(MATCH_RATE_STATS, source="player_match")
    validate_stats(CUMULATIVE_STATS, source="player_match")
    for variant in DEFCON_VARIANTS:
        validate_stats(variant.stats)
    defcon_counts = ",\n        ".join(
        f'{opta_count_sql(variant.stats, "o")} AS {variant.count}'
        for variant in DEFCON_VARIANTS
    )
    validate_stats(PENALTY_COMPONENT_STATS)
    pen_attempts = " + ".join(
        f"coalesce(o.{stat}, 0)" for stat in PENALTY_COMPONENT_STATS
    )
    pen = f"{pen_attempts} AS pen_attempts"
    player_pen_std = "sum(a.pen_attempts) OVER season_to_date"
    penalty_exposure = penalty_exposure_sql(
        player_pen_std,
        "a.team_pen_attempts_season_to_date",
        "a.team_matches_season_to_date",
    )
    anchor = per90_column_name(ANCHOR_STAT, rolling_window)
    defcon_windowed = ",\n    ".join(
        _defcon_windowed_sql(variant, rolling_window)
        for variant in DEFCON_VARIANTS
    )
    rates = ",\n        ".join(
        f"90.0 * sum(a.{stat}) OVER form "
        f"/ nullif(sum(CASE WHEN a.{stat} IS NOT NULL THEN a.minutes END) "
        f"OVER form, 0) AS {per90_column_name(stat, rolling_window)}"
        for stat in RATE_STATS
    )
    # Season-scoped and offset by one, so a booking counts towards every
    # later match in the season but never towards its own.
    # An empty window is the season's first appearance, where the running
    # total genuinely is zero. A window holding appearances but no
    # non-null value is a source that filed nothing, which is not the
    # same thing and must stay null -- a zero there reads as a clean
    # disciplinary record.
    totals = ",\n    ".join(
        f"CASE WHEN count(*) OVER season_to_date > 0 "
        f"AND count(a.{stat}) OVER season_to_date = 0 THEN NULL "
        f"ELSE coalesce(sum(a.{stat}) OVER season_to_date, 0) END "
        f"AS {cumulative_column_name(stat)}"
        for stat in CUMULATIVE_STATS
    )
    stat_columns = ",\n            ".join(
        [f"o.{stat}" for stat in OPTA_RATE_STATS]
        + [f"f.{stat}" for stat in FPL_RATE_STATS]
        + [
            f"m.{stat}"
            for stat in dict.fromkeys((*MATCH_RATE_STATS, *CUMULATIVE_STATS))
        ]
    )
    std_frame = (
        "ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW"
        if inclusive
        else "ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING"
    )
    appearances = f"""
appearances AS (
    SELECT
        m.season,
        m.gw,
        m.element,
        m.opponent,
        m.is_home,
        m.kickoff_time,
        m.minutes,
        m.total_points,
        {rolling_identity_sql()} AS rolling_identity,
        {defcon_counts},
        {pen},
        tp.team_pen_attempts_season_to_date,
        tp.team_matches_season_to_date,
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
    -- The club he turned out for that week, not the club the FCI
    -- snapshot ends the season with. Opposition is in the join for the
    -- same reason it is in points.py's: without it a double gameweek
    -- matches both of the club's fixtures and fans the row out.
    LEFT JOIN player_week AS pw
        ON  pw.season  = m.season
        AND pw.gw      = m.gw
        AND pw.element = m.element
    LEFT JOIN fpl_team_id AS opp_id
        ON  opp_id.season  = m.season
        AND opp_id.team_id = m.opponent
    LEFT JOIN team_penalty_form AS tp
        ON  tp.season     = m.season
        AND tp.gw         = m.gw
        AND tp.team       = pw.team
        AND tp.opposition = opp_id.team
    WHERE m.minutes > 0
)"""
    if inclusive:
        return f"""
WITH {appearances}
SELECT
    a.season,
    a.gw,
    a.element,
    a.opponent,
    a.is_home,
    a.kickoff_time,
    -- Selected, not just windowed on: the forward path as-of joins an
    -- unplayed fixture back to the player's last appearance and has to
    -- match on the same key the window partitions by, or a player with
    -- no appearance yet this season matches nothing.
    a.rolling_identity,
    a.minutes,
    a.total_points,
    {rates},
    {totals},
    {defcon_windowed},
    {penalty_exposure} AS {PENALTY_EXPOSURE_COLUMN},
    CASE WHEN {anchor} IS NULL THEN 1.0 ELSE 0.0 END AS {NO_FORM_COLUMN},
    count(*) OVER form AS form_matches,
    sum(a.minutes) OVER form AS form_minutes,
    date_diff('day', max(a.kickoff_time) OVER form, a.kickoff_time)
        AS days_since_last_appearance
FROM appearances AS a
WINDOW
    form AS (
        PARTITION BY a.rolling_identity
        ORDER BY a.kickoff_time
        {window_frame(rolling_window, inclusive)}
    ),
    season_to_date AS (
        PARTITION BY a.rolling_identity, a.season
        ORDER BY a.kickoff_time
        {std_frame}
    )
"""
    # The defcon columns are windowed exactly like the per-90 rates, so
    # they are computed in the same CTE and carried by the same as-of
    # join. Emitting them on the inclusive view alone would leave the
    # forward path working and training unable to bind at all.
    rate_names = [
        per90_column_name(stat, rolling_window) for stat in RATE_STATS
    ] + [
        name
        for variant in DEFCON_VARIANTS
        for name in variant.form_columns(rolling_window)
    ]
    carried_rates = ",\n    ".join(f"p.{name}" for name in rate_names)
    # Season-scoped: a player whose last appearance was last season
    # starts this one on nil rather than inheriting its closing tally,
    # and so does one who has not appeared at all. A null carried through
    # from a matched row is left alone -- it means the source filed no
    # cards, which the inner total already distinguishes from zero.
    carried_totals = ",\n    ".join(
        f"CASE WHEN p.season IS NULL OR p.season <> f.season THEN 0 "
        f"ELSE p.{name} END AS {name}"
        for name in (cumulative_column_name(stat) for stat in CUMULATIVE_STATS)
    )
    return f"""
WITH {appearances},
form AS (
    SELECT
        a.rolling_identity,
        a.season,
        a.kickoff_time,
        {rates},
        {totals},
        {defcon_windowed},
        {penalty_exposure} AS {PENALTY_EXPOSURE_COLUMN},
        count(*) OVER form AS form_matches,
        sum(a.minutes) OVER form AS form_minutes
    FROM appearances AS a
    WINDOW
        form AS (
            PARTITION BY a.rolling_identity
            ORDER BY a.kickoff_time
            {window_frame(rolling_window, True)}
        ),
        season_to_date AS (
            PARTITION BY a.rolling_identity, a.season
            ORDER BY a.kickoff_time
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )
),
fixtures AS (
    SELECT
        m.season,
        m.gw,
        m.element,
        m.opponent,
        m.is_home,
        m.kickoff_time,
        m.minutes,
        m.total_points,
        {rolling_identity_sql()} AS rolling_identity
    FROM player_match AS m
    LEFT JOIN player_season AS s
        ON s.season = m.season AND s.element = m.element
)
SELECT
    f.season,
    f.gw,
    f.element,
    f.opponent,
    f.is_home,
    f.kickoff_time,
    f.rolling_identity,
    f.minutes,
    f.total_points,
    {carried_rates},
    {carried_totals},
    -- Not season-guarded, unlike the card totals above. A card count is
    -- a running tally that resets; penalty duty is a standing role that
    -- does not, so an early-season row is better served by last
    -- season's settled figures than by a zero.
    p.{PENALTY_EXPOSURE_COLUMN},
    CASE WHEN p.{anchor} IS NULL THEN 1.0 ELSE 0.0 END AS {NO_FORM_COLUMN},
    coalesce(p.form_matches, 0) AS form_matches,
    p.form_minutes,
    date_diff('day', p.kickoff_time, f.kickoff_time)
        AS days_since_last_appearance
FROM fixtures AS f
ASOF LEFT JOIN form AS p
    ON  f.rolling_identity = p.rolling_identity
    AND f.kickoff_time > p.kickoff_time
"""


def register_match_form(
    connection: "DuckDBPyConnection",
    rolling_window: int = ROLLING_WINDOW,
    inclusive: bool = False,
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
    inclusive : bool, optional
        When True, register ``player_match_form_inclusive`` built on the
        inclusive frame instead of ``player_match_form``.
    """
    # opta_match is aliased so this module reads one name, whether the
    # bridge view is renamed later or swapped for a materialised table.
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {_VIEW_NAME}_opta AS "
        "SELECT * FROM opta_match"
    )
    # The penalty exposure feature is the one cross-player quantity here:
    # a player's share of his club's penalties needs his team-mates'
    # attempts, which is a team-level aggregation and lives there.
    register_team_penalty_form(connection)
    view = _INCLUSIVE_VIEW_NAME if inclusive else _VIEW_NAME
    connection.execute(
        f"CREATE OR REPLACE TEMP VIEW {view} AS "
        f"{form_sql(rolling_window, inclusive)}"
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
