"""Resolve FPL players to Transfermarkt players, once, at person level.

FPL and Transfermarkt share no identifier, so the bridge has to be built
from what both publish about a person: their name, their date of birth and
the clubs they have played for. Date of birth does the heavy lifting -- two
Premier League players sharing a birthday *and* a similar name is
vanishingly rare -- which demotes name similarity from "is this the same
person" to "confirm this is the same person".

The result is one row per matched ``player_code``. Identity is
time-invariant, so it is resolved once and the time-varying Transfermarkt
data (market values, transfers) joins through it by date afterwards.

The waterfall, in order. A ``player_code`` or ``tm_player_id`` claimed by
an earlier rung is never offered to a later one:

0. ``override``         -- the committed CSV, which always wins
1. ``dob_exact_name``   -- date of birth plus an exact normalised name
2. ``dob_fuzzy_name``   -- date of birth plus a similar full name
3. ``dob_surname``      -- date of birth plus a similar surname
4. ``club_exact_name``  -- no date of birth either side: exact name + club
5. ``exact_name``       -- an exact name unique on both sides, no DOB used
6. ``exact_name_dob_conflict`` -- as above, but the two dates disagree

The last two carry no date-of-birth evidence at all, so they lean entirely
on the name being the only one of its kind on both sides. The sixth is
split out because a shared name over two different birthdays is the
likeliest place for a wrong match to hide; query it to audit.

Within a rung, a player with two or more surviving candidates is left
unmatched rather than guessed at. Everything unmatched lands in the
candidate report, pre-filled with its best Transfermarkt candidates in the
override file's own schema, so closing the gap is a paste rather than a
Transfermarkt search.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import DATA_FOLDER
from fantasy_football.storage.tables import TM_PLAYER_MAP

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

MAPPINGS_FOLDER = Path(__file__).parent / "mappings"
OVERRIDES_PATH = MAPPINGS_FOLDER / "tm_player_overrides.csv"
CLUBS_PATH = MAPPINGS_FOLDER / "tm_clubs.csv"

# The report is regenerated every run, so it lives with the derived data
# rather than beside the two files a human maintains.
REPORT_PATH = DATA_FOLDER / "tm_player_map_candidates.csv"

# Jaro-Winkler, not Levenshtein: it weights a shared prefix, which is what
# distinguishes a transliteration from a different person.
NAME_SIMILARITY_THRESHOLD = 0.85

# How many Transfermarkt candidates each unmatched player gets offered.
REPORT_CANDIDATES = 3

OVERRIDE_COLUMNS = ("player_code", "tm_player_id", "fpl_name", "tm_name")


def _normalised_name(column: str) -> str:
    """Return SQL normalising a name column for comparison.

    Lowercased, accents stripped and every run of punctuation -- including
    the underscores in Vaastav-era names -- collapsed to a single space.
    """
    return (
        f"trim(regexp_replace(lower(strip_accents({column})), "
        "'[^a-z0-9]+', ' ', 'g'))"
    )


def _surname(expression: str) -> str:
    """Return SQL taking the last token of an already-normalised name."""
    return f"regexp_extract({expression}, '([a-z0-9]+)$', 1)"


def _normalised_club(column: str) -> str:
    """Return SQL normalising a club name for comparison.

    Transfermarkt is inconsistent about the ``FC``/``AFC`` suffix, so both
    are dropped. That is what lets clubs whose two names already agree
    match without needing a row in the club map at all.
    """
    padded = f"' ' || {_normalised_name(column)} || ' '"
    return f"trim(regexp_replace({padded}, ' (fc|afc) ', ' ', 'g'))"


def load_club_map(path: Path | None = None) -> pl.DataFrame:
    """Read the hand-maintained Transfermarkt-to-FPL club map.

    Parameters
    ----------
    path : Path | None, optional
        The CSV to read. Defaults to the committed ``CLUBS_PATH``.

    Returns
    -------
    pl.DataFrame
        Columns ``tm_club`` and ``fpl_club``.
    """
    frame = pl.read_csv(path or CLUBS_PATH)
    return frame.select(
        pl.col("tm_club").cast(pl.Utf8), pl.col("fpl_club").cast(pl.Utf8)
    )


def load_overrides(path: Path | None = None) -> pl.DataFrame:
    """Read the hand-maintained player overrides, checking they are 1:1.

    The file is edited by hand, so a duplicate on either side is a matter
    of when rather than if -- and a duplicate would fan out every
    ``player_week`` row it touches, silently inflating anything trained on
    the join.

    Parameters
    ----------
    path : Path | None, optional
        The CSV to read. Defaults to the committed ``OVERRIDES_PATH``.

    Returns
    -------
    pl.DataFrame
        Columns ``player_code``, ``tm_player_id``, ``fpl_name``,
        ``tm_name``.

    Raises
    ------
    ValueError
        If a ``player_code`` or a ``tm_player_id`` appears twice.
    """
    frame = pl.read_csv(
        path or OVERRIDES_PATH,
        schema_overrides={"player_code": pl.Int64, "tm_player_id": pl.Utf8},
    )
    frame = frame.select(OVERRIDE_COLUMNS)
    for column in ("player_code", "tm_player_id"):
        duplicates = (
            frame.filter(pl.col(column).is_duplicated())[column]
            .unique()
            .to_list()
        )
        if duplicates:
            raise ValueError(
                f"Player override file maps {column} more than once: "
                f"{sorted(duplicates)}. The map must be one-to-one."
            )
    return frame


def _temp_table(
    connection: "duckdb.DuckDBPyConnection",
    name: str,
    frame: pl.DataFrame,
    select: str = "*",
) -> None:
    """Materialise a frame as a temp table, unregistering it either way.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    name : str
        The temp table to create or replace.
    frame : pl.DataFrame
        The rows to materialise.
    select : str, optional
        The select list to read the frame through. Defaults to ``*``.
    """
    view = f"{name}_input"
    connection.register(view, frame.to_arrow())
    try:
        connection.execute(
            f"CREATE OR REPLACE TEMP TABLE {name} AS "
            f"SELECT {select} FROM {view}"
        )
    finally:
        connection.unregister(view)


def _register_sources(
    connection: "duckdb.DuckDBPyConnection", club_map: pl.DataFrame
) -> None:
    """Build the temp tables both sides of the waterfall read from.

    Four of them: a person and a club set for each side. Names and clubs
    arrive already normalised, so each rung is a plain join.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    club_map : pl.DataFrame
        The Transfermarkt-to-FPL club map.
    """
    _temp_table(
        connection,
        "tm_map_club",
        club_map,
        select=(
            f"{_normalised_club('tm_club')} AS tm_club_norm, "
            f"{_normalised_club('fpl_club')} AS fpl_club_norm"
        ),
    )
    # max() ignores nulls, so one observation of a person-level column in
    # any season supplies every season -- the same propagation
    # player_season does on read.
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE tm_map_fpl AS
        WITH person AS (
            SELECT
                player_code,
                max(birth_date) AS birth_date,
                max(first_name) AS first_name,
                max(second_name) AS second_name
            FROM player_season
            WHERE player_code IS NOT NULL
            GROUP BY player_code
        )
        SELECT
            player_code,
            birth_date,
            trim(coalesce(first_name, '') || ' ' || coalesce(second_name, ''))
                AS fpl_name,
            {_normalised_name("first_name || ' ' || second_name")}
                AS name_norm,
            {_surname(_normalised_name("second_name"))} AS surname_norm
        FROM person
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE tm_map_fpl_club AS
        SELECT DISTINCT
            ps.player_code,
            {_normalised_club("pw.team")} AS club_norm
        FROM player_week AS pw
        JOIN player_season AS ps
            ON ps.season = pw.season AND ps.element = pw.element
        WHERE ps.player_code IS NOT NULL AND pw.team IS NOT NULL
        """
    )
    connection.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE tm_map_tm AS
        SELECT
            tm_player_id,
            dob_date,
            name AS tm_name,
            player_link AS tm_player_link,
            {_normalised_name("name")} AS name_norm,
            {_surname(_normalised_name("name"))} AS surname_norm
        FROM tm_player
        """
    )
    # A club reaches FPL's spelling through the map where the two names
    # diverge, and through normalisation alone where they do not.
    connection.execute(
        """
        CREATE OR REPLACE TEMP TABLE tm_map_tm_club AS
        WITH raw AS (
            SELECT tm_player_id, team AS club FROM tm_player
            UNION
            SELECT tm_player_id, last_club FROM tm_player
            UNION
            SELECT tm_player_id, joined_club FROM tm_transfer
            UNION
            SELECT tm_player_id, left_club FROM tm_transfer
        ),
        normalised AS (
            SELECT tm_player_id, {club} AS club_norm
            FROM raw
            WHERE club IS NOT NULL
        )
        SELECT DISTINCT
            n.tm_player_id,
            coalesce(m.fpl_club_norm, n.club_norm) AS club_norm
        FROM normalised AS n
        LEFT JOIN tm_map_club AS m ON m.tm_club_norm = n.club_norm
        """.format(club=_normalised_club("club"))
    )


def _claim(
    connection: "duckdb.DuckDBPyConnection", matches: pl.DataFrame
) -> None:
    """Refresh the temp table of already-matched ids the rungs exclude."""
    _temp_table(
        connection,
        "tm_map_claimed",
        matches.select("player_code", "tm_player_id"),
    )


# Every rung is this one statement: pair the sides on the rung's own
# predicate, then keep only pairings that are the sole one on both sides.
_RUNG_QUERY = """
WITH pairs AS (
    SELECT
        f.player_code,
        t.tm_player_id,
        f.fpl_name,
        t.tm_name,
        {score} AS match_score
    FROM tm_map_fpl AS f
    JOIN tm_map_tm AS t ON {predicate}
    WHERE f.player_code NOT IN (SELECT player_code FROM tm_map_claimed)
      AND t.tm_player_id NOT IN (SELECT tm_player_id FROM tm_map_claimed)
),
unambiguous AS (
    SELECT * FROM pairs
    WHERE player_code IN (
        SELECT player_code FROM pairs GROUP BY player_code HAVING count(*) = 1
    )
    AND tm_player_id IN (
        SELECT tm_player_id FROM pairs
        GROUP BY tm_player_id HAVING count(*) = 1
    )
)
SELECT
    player_code,
    tm_player_id,
    '{rung}' AS match_rung,
    match_score,
    fpl_name,
    tm_name
FROM unambiguous
"""

# Two people sharing a birthday go to the human whatever their names
# score, exact matches included. Claimed ids do not count towards the
# crowd, so an override resolving one of a pair unblocks the other.
_SOLE_ON_DOB = """
    NOT EXISTS (
        SELECT 1 FROM tm_map_tm AS other
        WHERE other.dob_date = f.birth_date
          AND other.tm_player_id <> t.tm_player_id
          AND other.tm_player_id NOT IN (
              SELECT tm_player_id FROM tm_map_claimed
          )
    )
    AND NOT EXISTS (
        SELECT 1 FROM tm_map_fpl AS other
        WHERE other.birth_date = t.dob_date
          AND other.player_code <> f.player_code
          AND other.player_code NOT IN (
              SELECT player_code FROM tm_map_claimed
          )
    )
"""

# Both sides know a date of birth and they disagree. An exact name still
# matches on it, but under its own rung name so the weakest evidence in
# the map stays queryable.
_DOB_CONFLICT = """
    (
        f.birth_date IS NOT NULL
        AND t.dob_date IS NOT NULL
        AND f.birth_date <> t.dob_date
    )
"""

_SHARED_CLUB = """
    EXISTS (
        SELECT 1
        FROM tm_map_fpl_club AS fc
        JOIN tm_map_tm_club AS tc ON tc.club_norm = fc.club_norm
        WHERE fc.player_code = f.player_code
          AND tc.tm_player_id = t.tm_player_id
    )
"""


@dataclass(frozen=True)
class Rung:
    """One rung of the waterfall, as the SQL that drives it.

    Attributes
    ----------
    name : str
        The value stored in ``match_rung``.
    predicate : str
        The join condition between the two sides.
    score : str
        The expression stored in ``match_score``.
    """

    name: str
    predicate: str
    score: str


def _rungs(threshold: float) -> tuple[Rung, ...]:
    """Return the automatic rungs, in the order they are tried."""
    full = "jaro_winkler_similarity(f.name_norm, t.name_norm)"
    surname = "jaro_winkler_similarity(f.surname_norm, t.surname_norm)"
    same_dob = f"f.birth_date = t.dob_date AND {_SOLE_ON_DOB}"
    cutoff = float(threshold)
    return (
        Rung(
            "dob_exact_name",
            f"{same_dob} AND f.name_norm = t.name_norm",
            "1.0",
        ),
        Rung(
            "dob_fuzzy_name",
            f"{same_dob} AND {full} >= {cutoff}",
            full,
        ),
        Rung(
            "dob_surname",
            f"{same_dob} AND {surname} >= {cutoff}",
            surname,
        ),
        Rung(
            "club_exact_name",
            "(f.birth_date IS NULL OR t.dob_date IS NULL) "
            f"AND f.name_norm = t.name_norm AND {_SHARED_CLUB}",
            "1.0",
        ),
        Rung(
            "exact_name",
            f"f.name_norm = t.name_norm AND NOT {_DOB_CONFLICT}",
            "1.0",
        ),
        Rung(
            "exact_name_dob_conflict",
            f"f.name_norm = t.name_norm AND {_DOB_CONFLICT}",
            "1.0",
        ),
    )


def build_player_map(
    connection: "duckdb.DuckDBPyConnection",
    overrides: pl.DataFrame | None = None,
    club_map: pl.DataFrame | None = None,
    threshold: float = NAME_SIMILARITY_THRESHOLD,
) -> pl.DataFrame:
    """Resolve every FPL player_code to a Transfermarkt player it can.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection carrying the ``tm_*`` and identity tables.
    overrides : pl.DataFrame | None, optional
        Hand-written matches. Defaults to the committed override file.
    club_map : pl.DataFrame | None, optional
        The club map. Defaults to the committed club file.
    threshold : float, optional
        Minimum Jaro-Winkler similarity for the two fuzzy rungs.

    Returns
    -------
    pl.DataFrame
        One row per matched player, shaped to ``TM_PLAYER_MAP``.
    """
    overrides = load_overrides() if overrides is None else overrides
    club_map = load_club_map() if club_map is None else club_map
    _register_sources(connection, club_map)

    matches = _override_matches(connection, overrides)
    for rung in _rungs(threshold):
        _claim(connection, matches)
        found = connection.execute(
            _RUNG_QUERY.format(
                predicate=rung.predicate, score=rung.score, rung=rung.name
            )
        ).pl()
        logger.info("Rung %s matched %d players.", rung.name, found.height)
        matches = pl.concat([matches, _shape(found)])
    return matches.sort("player_code")


def _override_matches(
    connection: "duckdb.DuckDBPyConnection", overrides: pl.DataFrame
) -> pl.DataFrame:
    """Return the override rows, shaped and with a null score.

    The names in the file are the maintainer's own notes, so the stored
    ones are re-read from each side instead -- a stale comment column must
    not become the name the rest of the pipeline sees.
    """
    if overrides.is_empty():
        return _shape(pl.DataFrame(schema=TM_PLAYER_MAP.schema))
    _temp_table(connection, "tm_map_override", overrides)
    found = connection.execute(
        """
        SELECT
            o.player_code,
            o.tm_player_id,
            'override' AS match_rung,
            CAST(NULL AS DOUBLE) AS match_score,
            coalesce(f.fpl_name, o.fpl_name) AS fpl_name,
            coalesce(t.tm_name, o.tm_name) AS tm_name
        FROM tm_map_override AS o
        LEFT JOIN tm_map_fpl AS f ON f.player_code = o.player_code
        LEFT JOIN tm_map_tm AS t ON t.tm_player_id = o.tm_player_id
        """
    ).pl()
    return _shape(found)


def _shape(frame: pl.DataFrame) -> pl.DataFrame:
    """Narrow a rung's output to the stored column order and dtypes."""
    return TM_PLAYER_MAP.coerce(TM_PLAYER_MAP.conform(frame))


def unmatched_report(
    connection: "duckdb.DuckDBPyConnection",
    matches: pl.DataFrame,
    candidates: int = REPORT_CANDIDATES,
) -> pl.DataFrame:
    """Return unmatched FPL players and their best candidates.

    The rows carry the override file's own columns first, so a correct
    candidate is pasted straight across, with the evidence that produced
    it alongside for eyeballing -- including the Transfermarkt page, so a
    doubtful one is one click away.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection, after ``build_player_map`` has run.
    matches : pl.DataFrame
        What ``build_player_map`` returned.
    candidates : int, optional
        Candidates offered per unmatched player.

    Returns
    -------
    pl.DataFrame
        One row per candidate, best first within each player.
    """
    _claim(connection, matches)
    return connection.execute(
        f"""
        WITH ranked AS (
            SELECT
                f.player_code,
                t.tm_player_id,
                f.fpl_name,
                t.tm_name,
                t.tm_player_link,
                f.birth_date,
                t.dob_date,
                jaro_winkler_similarity(f.name_norm, t.name_norm) AS score,
                row_number() OVER (
                    PARTITION BY f.player_code
                    ORDER BY jaro_winkler_similarity(f.name_norm, t.name_norm)
                        DESC, t.tm_player_id
                ) AS rank
            FROM tm_map_fpl AS f
            LEFT JOIN tm_map_tm AS t
                ON t.tm_player_id NOT IN (
                    SELECT tm_player_id FROM tm_map_claimed
                )
            WHERE f.player_code NOT IN (SELECT player_code FROM tm_map_claimed)
        )
        SELECT
            player_code,
            tm_player_id,
            fpl_name,
            tm_name,
            tm_player_link,
            birth_date AS fpl_birth_date,
            dob_date AS tm_dob_date,
            score AS name_similarity
        FROM ranked
        WHERE rank <= {int(candidates)}
        ORDER BY player_code, rank
        """
    ).pl()


def store_player_map(
    connection: "duckdb.DuckDBPyConnection", matches: pl.DataFrame
) -> None:
    """Replace the stored map wholesale.

    Wholesale because the committed CSV is the only durable hand-made
    half: nothing accumulates in the table that a rebuild cannot
    reproduce, so it can never drift from the files that made it.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    matches : pl.DataFrame
        What ``build_player_map`` returned.
    """
    TM_PLAYER_MAP.replace_all(connection, matches)


def refresh_player_map(
    connection: "duckdb.DuckDBPyConnection",
    report_path: Path | None = None,
) -> pl.DataFrame:
    """Rebuild the stored map and write the candidate report beside it.

    The report goes to the gitignored data folder rather than the mappings
    folder: a stale report must never be mistaken for a real mapping.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    report_path : Path | None, optional
        Where to write the candidate report. Defaults to ``REPORT_PATH``.

    Returns
    -------
    pl.DataFrame
        The stored matches.
    """
    matches = build_player_map(connection)
    store_player_map(connection, matches)
    report = unmatched_report(connection, matches)
    unmatched = (
        report["player_code"].n_unique() if not report.is_empty() else 0
    )
    path = report_path or REPORT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    report.write_csv(path)
    logger.info(
        "Matched %d players; %d unmatched. Candidates written to %s",
        matches.height,
        unmatched,
        path,
    )
    for rung, count in (
        matches["match_rung"].value_counts().sort("match_rung").iter_rows()
    ):
        logger.info("  %s: %d", rung, count)
    return matches
