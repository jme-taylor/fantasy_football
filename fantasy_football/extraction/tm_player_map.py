import logging
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel, Field

from fantasy_football.constants import DATA_FOLDER
from fantasy_football.storage.tables import TM_PLAYER_MAP

if TYPE_CHECKING:
    import duckdb

logger = logging.getLogger(__name__)

MAPPINGS_FOLDER = Path(__file__).parent / "mappings"
OVERRIDES_PATH = MAPPINGS_FOLDER / "tm_player_overrides.csv"
CLUBS_PATH = MAPPINGS_FOLDER / "tm_clubs.csv"
CANDIDATE_PATH = DATA_FOLDER / "tm_player_map_candidates.csv"

NAME_SIMILARITY_THRESHOLD = 0.85

CANDIDATE_COUNT = 3

OVERRIDE_COLUMNS = ("player_code", "tm_player_id", "fpl_name", "tm_name")

PLAYER_POSITIONS = ("GK", "DEF", "MID", "FWD")


class TransferMarktMatchRule(BaseModel):
    """A single rule for matching FPL players to Transfermarkt players.

    For each matching rule, when a player is matched on that rule, the score
    is calculated and stored in the match_score column. The rule that was used
    to make the match is stored in the match_rule column.
    """

    name: str = Field(description="The value stored in the match_rule column.")
    predicate: str = Field(
        description="The join condition between the two sides."
    )
    score: str = Field(
        description="The expression stored in the match_score column."
    )


def create_match_rules(threshold: float) -> tuple[TransferMarktMatchRule, ...]:
    """Create the match rules for the waterfall.

    These will be executed in order, and the first rule that matches will be
    used to match the player.

    Parameters
    ----------
    threshold: float
        The threshold for the fuzzy match rules.

    Returns
    -------
    tuple[TransferMarktMatchRule, ...]
        The match rules for the waterfall.
    """
    full = "jaro_winkler_similarity(f.name_norm, t.name_norm)"
    surname = "jaro_winkler_similarity(f.surname_norm, t.surname_norm)"
    same_dob = f"f.birth_date = t.dob_date AND {_SOLE_ON_DOB}"
    cutoff = float(threshold)
    return (
        TransferMarktMatchRule(
            name="dob_exact_name",
            predicate=f"{same_dob} AND f.name_norm = t.name_norm",
            score="1.0",
        ),
        TransferMarktMatchRule(
            name="dob_fuzzy_name",
            predicate=f"{same_dob} AND {full} >= {cutoff}",
            score=full,
        ),
        TransferMarktMatchRule(
            name="dob_surname",
            predicate=f"{same_dob} AND {surname} >= {cutoff}",
            score=surname,
        ),
        TransferMarktMatchRule(
            name="club_exact_name",
            predicate="(f.birth_date IS NULL OR t.dob_date IS NULL) "
            f"AND f.name_norm = t.name_norm AND {_SHARED_CLUB}",
            score="1.0",
        ),
        TransferMarktMatchRule(
            name="exact_name",
            predicate=f"f.name_norm = t.name_norm AND NOT {_DOB_CONFLICT}",
            score="1.0",
        ),
        TransferMarktMatchRule(
            name="exact_name_dob_conflict",
            predicate=f"f.name_norm = t.name_norm AND {_DOB_CONFLICT}",
            score="1.0",
        ),
    )


class TransferMarktPlayerMap:
    """Class for matching FPL players to Transfermarkt players."""

    def __init__(
        self,
        connection: "duckdb.DuckDBPyConnection",
        threshold: float = NAME_SIMILARITY_THRESHOLD,
        report_candidates: int = CANDIDATE_COUNT,
    ) -> None:
        """Hold the connection and the settings the waterfall runs under."""
        self.connection = connection
        self.threshold = threshold
        self.report_candidates = report_candidates
        self.match_rules = create_match_rules(threshold)

    def build_player_map(self) -> pl.DataFrame:
        """Match FPL players to Transfermarkt players using waterfall.

        This will always first check the overrides file that has been
        manually edited by a human, and then the rest of the candidates will
        go through the waterfall of match rules. The order of the match rules
        is determined by the create_match_rules function.

        Returns
        -------
        pl.DataFrame
            A DataFrame containing the matched FPL players and their Transfermarkt player IDs.
        """
        overrides = load_overrides()
        club_map = load_club_map()
        self._register_sources(club_map)

        player_matches = self._override_matches(overrides)

        for rule in self.match_rules:
            self._refresh_claimed_ids(player_matches)
            found = self.connection.execute(
                MATCH_RULE_QUERY.format(
                    predicate=rule.predicate,
                    score=rule.score,
                    match_rule=rule.name,
                )
            ).pl()
            logger.info(
                "Found %d matches using the %s rule", found.height, rule.name
            )
            player_matches = pl.concat([player_matches, _shape(found)])
        return player_matches.sort("player_code")

    def create_unmatched_report(self, player_matches: pl.DataFrame) -> int:
        """Create a report of unmatched players for manual review.

        First refreshes the claimed IDs table to ensure it is fresh. It
        then creates a report of the unmatched players, with the top N
        most similar players, as determined by the report_candidates class
        variable. The report is written to the CANDIDATE_PATH file.

        Parameters
        ----------
        player_matches: pl.DataFrame
            The player matches to create the report for.

        Returns
        -------
        int
            The number of unmatched players
        """
        self._refresh_claimed_ids(player_matches)
        unmatched_player_report = self.connection.execute(
            UNMATCHED_REPORT_QUERY.format(candidates=self.report_candidates)
        ).pl()
        unmatched_count = (
            unmatched_player_report["player_code"].n_unique()
            if not unmatched_player_report.is_empty()
            else 0
        )
        CANDIDATE_PATH.parent.mkdir(parents=True, exist_ok=True)
        unmatched_player_report.write_csv(CANDIDATE_PATH)
        return unmatched_count

    def refresh_player_map(self) -> pl.DataFrame:
        """Refresh the player map.

        First builds the player matches, and then updates the TM_PLAYER_MAP
        table with all matches. It then creates and saves the unmatched
        player report.

        Returns
        -------
        pl.DataFrame
            The player matches.
        """
        player_matches = self.build_player_map()
        TM_PLAYER_MAP.replace_all(self.connection, player_matches)
        unmatched_count = self.create_unmatched_report(player_matches)
        logger.info(
            "Matched %d players; %d unmatched",
            player_matches.height,
            unmatched_count,
        )
        for match_rule, count in (
            player_matches["match_rule"]
            .value_counts()
            .sort("match_rule")
            .iter_rows()
        ):
            logger.info(" %s: %d", match_rule, count)
        return player_matches

    def _register_sources(self, club_map: pl.DataFrame) -> None:
        """Register all sources required for the player mapping.

        This will register the club map, the FPL player map, the Transfermarkt
        player map and the Transfermarkt club map.

        Parameters
        ----------
        club_map: pl.DataFrame
            The club map to register.
        """
        _temp_table(
            self.connection,
            "tm_map_club",
            club_map,
            select=(
                f"{_normalised_club('tm_club')} AS tm_club_norm, {_normalised_club('fpl_club')} AS fpl_club_norm"
            ),
        )
        self.connection.execute(
            TM_MAP_FPL_QUERY.format(
                valid_positions=_sql_tuple(PLAYER_POSITIONS),
                name_normalised=_normalised_name(
                    "first_name || ' ' || second_name"
                ),
                surname_normalised=_surname(_normalised_name("second_name")),
            )
        )
        self.connection.execute(
            TM_MAP_FPL_CLUB_QUERY.format(
                normalised_club=_normalised_club("pw.team")
            )
        )
        self.connection.execute(
            TM_MAP_TM_QUERY.format(
                name_normalised=_normalised_name("name"),
                surname_normalised=_surname(_normalised_name("name")),
            )
        )
        self.connection.execute(
            TM_MAP_TM_CLUB_QUERY.format(
                club_normalised=_normalised_club("club")
            )
        )

    def _override_matches(self, overrides: pl.DataFrame) -> pl.DataFrame:
        if overrides.is_empty():
            return _shape(pl.DataFrame(schema=TM_PLAYER_MAP.schema))
        _temp_table(self.connection, "tm_map_override", overrides)
        found = self.connection.execute(FOUND_OVERRIDES_QUERY).pl()
        return _shape(found)

    def _refresh_claimed_ids(self, player_matches: pl.DataFrame) -> None:
        """Refresh the claimed IDs table.

        This stops other rules from matching the player.

        Parameters
        ----------
        player_matches: pl.DataFrame
            The player matches to refresh the claimed IDs table with.
        """
        _temp_table(
            self.connection,
            "tm_map_claimed",
            player_matches.select("player_code", "tm_player_id"),
        )


def load_club_map() -> pl.DataFrame:
    """Read the hand-maintained Transfermarkt-to-FPL club map.

    Returns
    -------
    pl.DataFrame
        Columns ``tm_club`` and ``fpl_club``.
    """
    frame = pl.read_csv(CLUBS_PATH)
    return frame.select(
        pl.col("tm_club").cast(pl.Utf8), pl.col("fpl_club").cast(pl.Utf8)
    )


def load_overrides() -> pl.DataFrame:
    """Read the hand-maintained player overrides, checking they are 1:1.

    The file is edited by hand, so a duplicate on either side is likely due
    to human error. A duplicate would fan out every ``player_week`` row it
    touches, silently inflating anything trained on the join.

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
        OVERRIDES_PATH,
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


def _normalised_name(column: str) -> str:
    """Return SQL normalising a name column for comparison.

    Lowercased, accents stripped and every run of punctuation -- including
    the underscores in Vaastav-era names -- collapsed to a single space.

    Parameters
    ----------
    column : str
        The name column in the table to normalise.

    Returns
    -------
    str
        The SQL to normalise the column.
    """
    return (
        f"trim(regexp_replace(lower(strip_accents({column})), "
        "'[^a-z0-9]+', ' ', 'g'))"
    )


def _surname(expression: str) -> str:
    """Return SQL taking the last token of an already-normalised name.

    Parameters
    ----------
    expression: str
        The SQL to extract the surname from.

    Returns
    -------
    str
        The SQL to extract the surname from.
    """
    return f"regexp_extract({expression}, '([a-z0-9]+)$', 1)"


def _normalised_club(column: str) -> str:
    """Return SQL normalising a club name for comparison.

    Transfermarkt is inconsistent about the ``FC``/``AFC`` suffix, so both
    are dropped. That is what lets clubs whose two names already agree
    match without needing a row in the club map at all.

    Parameters
    ----------
    column: str
        The club column in the table to normalise.

    Returns
    -------
    str
        The SQL to normalise the club column.
    """
    padded = f"' ' || {_normalised_name(column)} || ' '"
    return f"trim(regexp_replace({padded}, ' (fc|afc) ', ' ', 'g'))"


def _sql_tuple(values: tuple[str, ...]) -> str:
    """Render a tuple of labels as a SQL ``IN`` list.

    Parameters
    ----------
    values: tuple[str, ...]
        The values to render as a SQL ``IN`` list.

    Returns
    -------
    str
        The SQL to render the values as a ``IN`` list.
    """
    joined = ", ".join(f"'{value}'" for value in values)
    return f"({joined})"


def _temp_table(
    connection: "duckdb.DuckDBPyConnection",
    table_name: str,
    frame: pl.DataFrame,
    select: str = "*",
) -> None:
    """Materialise a frame as a temp table, unregistering it either way.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    table_name : str
        The temp table to create or replace.
    frame : pl.DataFrame
        The rows to materialise.
    select : str, optional
        The select list to read the frame through. Defaults to ``*``.
    """
    view = f"{table_name}_input"
    connection.register(view, frame.to_arrow())
    try:
        connection.execute(
            f"CREATE OR REPLACE TEMP TABLE {table_name} AS "
            f"SELECT {select} FROM {view}"
        )
    finally:
        connection.unregister(view)


def _shape(frame: pl.DataFrame) -> pl.DataFrame:
    """Narrow a match rule's output to the stored column order and dtypes."""
    return TM_PLAYER_MAP.coerce(TM_PLAYER_MAP.conform(frame))


MATCH_RULE_QUERY = """
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
    '{match_rule}' AS match_rule,
    match_score,
    fpl_name,
    tm_name
FROM unambiguous
"""

UNMATCHED_REPORT_QUERY = """
WITH ranked AS (
    SELECT
        f.player_code,
        t.tm_player_id,
        f.fpl_name,
        t.tm_name,
        t.tm_player_link,
        f.player_weeks,
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
        AND f.player_weeks > 0
)
SELECT
    player_code,
    tm_player_id,
    fpl_name,
    tm_name,
    tm_player_link,
    player_weeks AS fpl_player_weeks,
    birth_date AS fpl_birth_date,
    dob_date AS tm_dob_date,
    score AS name_similarity
FROM ranked
WHERE rank <= {candidates}
ORDER BY player_code, rank
"""

# TODO(JT): Make this query name more verbose
TM_MAP_FPL_QUERY = """
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
    HAVING count(*) FILTER (
        WHERE position IN {valid_positions}
    ) > 0
),
weeks AS (
    SELECT ps.player_code, count(*) AS player_weeks
    FROM player_week AS pw
    JOIN player_season AS ps
        ON ps.season = pw.season AND ps.element = pw.element
    WHERE ps.player_code IS NOT NULL
    GROUP BY ps.player_code
)
SELECT
    person.player_code,
    coalesce(weeks.player_weeks, 0) AS player_weeks,
    birth_date,
    trim(coalesce(first_name, '') || ' ' || coalesce(second_name, ''))
        AS fpl_name,
    {name_normalised}
        AS name_norm,
    {surname_normalised} AS surname_norm
FROM person
LEFT JOIN weeks USING (player_code)
"""

# TODO(JT): Make this query name more verbose
TM_MAP_FPL_CLUB_QUERY = """
CREATE OR REPLACE TEMP TABLE tm_map_fpl_club AS
SELECT DISTINCT
    ps.player_code,
    {normalised_club} AS club_norm
FROM player_week AS pw
JOIN player_season AS ps
    ON ps.season = pw.season AND ps.element = pw.element
WHERE ps.player_code IS NOT NULL AND pw.team IS NOT NULL
"""

# TODO(JT): Make this query name more verbose
TM_MAP_TM_QUERY = """
CREATE OR REPLACE TEMP TABLE tm_map_tm AS
SELECT
    tm_player_id,
    dob_date,
    name AS tm_name,
    player_link AS tm_player_link,
    {name_normalised} AS name_norm,
    {surname_normalised} AS surname_norm
FROM tm_player
"""

# TODO(JT): Make this query name more verbose
TM_MAP_TM_CLUB_QUERY = """
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
    SELECT tm_player_id, {club_normalised} AS club_norm
    FROM raw
    WHERE club IS NOT NULL
)
SELECT DISTINCT
    n.tm_player_id,
    coalesce(m.fpl_club_norm, n.club_norm) AS club_norm
FROM normalised AS n
LEFT JOIN tm_map_club AS m ON m.tm_club_norm = n.club_norm
"""

FOUND_OVERRIDES_QUERY = """
SELECT
    o.player_code,
    o.tm_player_id,
    'override' AS match_rule,
    CAST(NULL AS DOUBLE) AS match_score,
    coalesce(f.fpl_name, o.fpl_name) AS fpl_name,
    coalesce(t.tm_name, o.tm_name) AS tm_name
FROM tm_map_override AS o
LEFT JOIN tm_map_fpl AS f ON f.player_code = o.player_code
LEFT JOIN tm_map_tm AS t ON t.tm_player_id = o.tm_player_id
"""

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
