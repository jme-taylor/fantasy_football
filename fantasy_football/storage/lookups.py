"""Session-scoped views that bridge FCI fixtures to FPL fixtures.

``player_match`` identifies a fixture by ``(opponent, is_home)``, where
``opponent`` is the FPL team id. ``player_match_opta`` identifies one by
``match_id``, a club-name slug like
``25-26-prem-manchester-united-vs-arsenal``. Nothing joins the two, so
without a bridge the only honest join is at gameweek grain -- which
collapses the two legs of a double gameweek into one row.

This module registers four temporary views that supply the bridge:

``fci_slug_map``
    ``FCI_SLUG_TO_FPL`` and ``FPL_TEAM_ALIASES`` as a relation: one row
    per slug and FPL name the club has carried, beside its canonical
    ``club`` name.
``fpl_team_id``
    FPL team id <-> club name, per season. Derived from the data rather
    than hardcoded, by matching each ``player_match`` leg to its
    ``team_fixture`` row on the player's club and kickoff time.
``fci_club``
    ``fci_slug_map`` narrowed to the FPL name each season's
    ``team_fixture`` actually uses,
    so a club FPL renames (``"Ipswich"`` to ``"Ipswich Town"``) resolves
    to the name ``player_week`` and ``team_fixture`` carry that season.
``opta_match``
    Premier League ``player_match_opta`` rows carrying a resolved
    ``opponent`` and ``is_home``, so they join straight to
    ``player_match`` on its primary key.

With those in place the per-match join is one statement::

    SELECT m.*, o.tackles, o.interceptions
    FROM player_match AS m
    LEFT JOIN opta_match AS o USING (season, gw, element, opponent)

The views are TEMP: they live for the connection only and never land in
the database file, so they are invisible to the schema-drift check.

Two caveats hold regardless of how the join is keyed. The player's club
comes from ``player_week.team``, which is per-gameweek -- do not
substitute ``player_snapshot.team``, which holds the player's final club
and would resolve home/away wrongly for any January transfer. And FCI
occasionally files a rearranged fixture under a different gameweek than
FPL, leaving a small number of Opta rows with no ``player_match``
counterpart at all.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import FCI_SLUG_TO_FPL, FPL_TEAM_ALIASES

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# The club slugs sit between the competition token and the "-vs-"
# separator: ``25-26-prem-<home>-vs-<away>``.
_HOME_SLUG = r"^\d{2}-\d{2}-prem-(.+)-vs-(.+)$"

_FPL_TEAM_ID_SQL = """
CREATE OR REPLACE TEMP VIEW fpl_team_id AS
SELECT DISTINCT
    m.season,
    m.opponent    AS team_id,
    tf.opposition AS team
FROM player_match AS m
JOIN player_week AS w
    USING (season, gw, element)
JOIN team_fixture AS tf
    ON  tf.season       = m.season
    AND tf.gw           = m.gw
    AND tf.team         = w.team
    AND tf.is_home      = m.is_home
    AND tf.kickoff_time = m.kickoff_time
"""

# Either column of team_fixture names a club that season; reading both
# means a fixture seeded from one side still resolves its opponent.
_FCI_CLUB_SQL = """
CREATE OR REPLACE TEMP VIEW fci_club AS
WITH season_team AS (
    SELECT season, team FROM team_fixture
    UNION
    SELECT season, opposition FROM team_fixture
)
SELECT DISTINCT t.season, s.slug, s.club, s.team
FROM fci_slug_map AS s
JOIN season_team AS t
    ON t.team = s.team
"""

_OPTA_MATCH_SQL = f"""
CREATE OR REPLACE TEMP VIEW opta_match AS
WITH resolved AS (
    SELECT
        o.*,
        w.team      AS player_team,
        home.team   AS home_team,
        away.team   AS away_team
    FROM player_match_opta AS o
    JOIN player_week AS w
        USING (season, gw, element)
    LEFT JOIN fci_club AS home
        ON  home.season = o.season
        AND home.slug   = regexp_extract(o.match_id, '{_HOME_SLUG}', 1)
    LEFT JOIN fci_club AS away
        ON  away.season = o.season
        AND away.slug   = regexp_extract(o.match_id, '{_HOME_SLUG}', 2)
    WHERE o.competition = 'prem'
)
SELECT
    r.* EXCLUDE (player_team, home_team, away_team),
    r.player_team = r.home_team AS is_home,
    t.team_id                   AS opponent
FROM resolved AS r
JOIN fpl_team_id AS t
    ON  t.season = r.season
    AND t.team   = CASE
                       WHEN r.player_team = r.home_team THEN r.away_team
                       ELSE r.home_team
                   END
"""


def slug_frame() -> pl.DataFrame:
    """Return ``FCI_SLUG_TO_FPL`` and its aliases as a frame.

    Returns
    -------
    pl.DataFrame
        Columns ``slug``, ``club`` and ``team``: one row per known FCI
        slug and FPL name its club has carried, with ``club`` the
        canonical name shared by all of them.
    """
    rows = [
        (slug, club, team)
        for slug, club in FCI_SLUG_TO_FPL.items()
        for team in [club, *FPL_TEAM_ALIASES.get(club, [])]
    ]
    return pl.DataFrame(
        rows,
        schema={"slug": pl.Utf8, "club": pl.Utf8, "team": pl.Utf8},
        orient="row",
    )


def unknown_slugs(connection: "DuckDBPyConnection") -> list[str]:
    """Return Premier League club slugs absent from ``FCI_SLUG_TO_FPL``.

    A promoted club FCI publishes before the constant is updated would
    otherwise vanish from ``opta_match`` silently.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.

    Returns
    -------
    list[str]
        Sorted unmapped slugs; empty when the constant is complete.
    """
    found = connection.sql(
        f"""
        SELECT DISTINCT regexp_extract(match_id, '{_HOME_SLUG}', 1) AS slug
        FROM player_match_opta WHERE competition = 'prem'
        UNION
        SELECT DISTINCT regexp_extract(match_id, '{_HOME_SLUG}', 2) AS slug
        FROM player_match_opta WHERE competition = 'prem'
        """
    ).pl()["slug"]
    return sorted(set(found.to_list()) - set(FCI_SLUG_TO_FPL) - {""})


def unresolved_slugs(connection: "DuckDBPyConnection") -> list[str]:
    """Return ``season: slug`` pairs whose club has no FPL name that season.

    A known slug only resolves through ``fci_club`` when one of its names
    is in that season's ``team_fixture``. A club FPL renames, without the
    new name in ``FPL_TEAM_ALIASES``, would otherwise drop out of
    ``opta_match`` silently -- along with every opponent it faced.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection with the lookup views registered.

    Returns
    -------
    list[str]
        Sorted ``"<season>: <slug>"`` entries; empty when all resolve.
    """
    rows = connection.sql(
        f"""
        WITH used AS (
            SELECT DISTINCT season, regexp_extract(match_id, '{_HOME_SLUG}', 1) AS slug
            FROM player_match_opta WHERE competition = 'prem'
            UNION
            SELECT DISTINCT season, regexp_extract(match_id, '{_HOME_SLUG}', 2) AS slug
            FROM player_match_opta WHERE competition = 'prem'
        )
        SELECT u.season, u.slug
        FROM used AS u
        JOIN fci_slug_map AS s USING (slug)
        ANTI JOIN fci_club AS c USING (season, slug)
        GROUP BY ALL
        ORDER BY ALL
        """
    ).fetchall()
    return [f"{season}: {slug}" for season, slug in rows]


def register_lookups(connection: "DuckDBPyConnection") -> None:
    """Create the four fixture-bridge views.

    ``fci_slug_map``, ``fpl_team_id``, ``fci_club`` and ``opta_match``.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection. The views are temporary and last only as long
        as it does.
    """
    connection.register("fci_slug_map_frame", slug_frame())
    connection.execute(
        "CREATE OR REPLACE TEMP VIEW fci_slug_map AS "
        "SELECT * FROM fci_slug_map_frame"
    )
    connection.execute(_FPL_TEAM_ID_SQL)
    connection.execute(_FCI_CLUB_SQL)
    connection.execute(_OPTA_MATCH_SQL)

    missing = unknown_slugs(connection)
    if missing:
        logger.warning(
            "FCI publishes Premier League club slug(s) absent from "
            "FCI_SLUG_TO_FPL: %s. Their matches are missing from "
            "opta_match until the constant is updated.",
            ", ".join(missing),
        )
    unresolved = unresolved_slugs(connection)
    if unresolved:
        logger.warning(
            "FCI club slug(s) with no FPL name in that season's "
            "team_fixture: %s. FPL has likely renamed the club; add the "
            "new name to FPL_TEAM_ALIASES, or its matches are missing "
            "from opta_match.",
            ", ".join(unresolved),
        )
