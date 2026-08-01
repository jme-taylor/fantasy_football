"""Load FCI's per-fixture Opta stats into ``player_match_opta``.

FCI publishes one ``playermatchstats.csv`` per gameweek, covering every
competition its players appeared in -- Premier League, EFL Cup, and the
three European competitions. All of it is stored: midweek European
minutes are genuine rotation and fatigue signal. The ``competition``
column derived here is what makes the FPL filter explicit; any consumer
computing FPL minutes must filter to ``"prem"``.

Two source quirks are handled. The 2024-25 season stores files under
``playermatchstats/`` while later seasons use ``By Gameweek/``. And
column dtypes drift between gameweek files -- a column that reads as
Int64 in one file reads as String in another, and the wholly-null
distance columns always read as String -- so the frame is cast to the
table's declared schema rather than trusting per-file inference.
"""

import logging
import re
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import CURRENT_SEASON
from fantasy_football.extraction.fci import FciExtractor
from fantasy_football.extraction.seasons import (
    season_long_to_short,
    season_short_to_long,
)
from fantasy_football.storage.tables import PLAYER_MATCH_OPTA

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# Competition tokens observed in FCI match_ids. An unrecognised token is
# stored rather than dropped, but logged so a new competition is visible.
KNOWN_COMPETITIONS: frozenset[str] = frozenset(
    {"prem", "efl", "champions", "europa", "conference"}
)

# match_ids look like ``25-26-prem-manchester-united-v-arsenal``.
_COMPETITION_RE = re.compile(r"^\d{2}-\d{2}-([a-z]+)-")

# Both repo layouts, with the gameweek number captured.
_MATCHSTATS_PATH_RE = re.compile(
    r"^data/(\d{4}-\d{4})/(?:By Gameweek|playermatchstats)/GW(\d+)/"
    r"playermatchstats\.csv$"
)


def parse_competition(match_id: str) -> str | None:
    """Return the competition token embedded in an FCI match_id.

    Parameters
    ----------
    match_id : str
        An FCI fixture slug, e.g.
        ``"25-26-prem-manchester-united-v-arsenal"``.

    Returns
    -------
    str | None
        The competition token, or None when the slug does not match the
        expected shape.
    """
    match = _COMPETITION_RE.match(match_id)
    return match.group(1) if match else None


def add_competition(frame: pl.DataFrame) -> pl.DataFrame:
    """Add a ``competition`` column parsed from ``match_id``.

    Unrecognised tokens are kept, not dropped -- a competition FCI has
    not carried before is still real data -- but they are logged.

    Parameters
    ----------
    frame : pl.DataFrame
        A frame carrying a ``match_id`` column.

    Returns
    -------
    pl.DataFrame
        The frame with a ``competition`` column added.
    """
    result = frame.with_columns(
        pl.col("match_id")
        .str.extract(r"^\d{2}-\d{2}-([a-z]+)-", 1)
        .alias("competition")
    )
    seen = {
        value
        for value in result["competition"].unique().to_list()
        if value is not None
    }
    unrecognised = seen - KNOWN_COMPETITIONS
    if unrecognised:
        logger.warning(
            "FCI match_ids carry unrecognised competition token(s): %s. "
            "Rows are stored as-is; add them to KNOWN_COMPETITIONS once "
            "confirmed.",
            ", ".join(sorted(unrecognised)),
        )
    return result


class FciMatchStatsLoader:
    """Download FCI per-fixture stats and store them by season."""

    def __init__(self, extractor: FciExtractor | None = None) -> None:
        """Initialise the loader.

        Parameters
        ----------
        extractor : FciExtractor | None, optional
            FCI extractor, reused for its GitHub client and CSV reader.
            Defaults to a new ``FciExtractor``.
        """
        self.extractor = extractor or FciExtractor()

    def _matchstats_tree(self) -> list[tuple[str, int, str]]:
        """Return every playermatchstats path as (season, gw, path).

        Returns
        -------
        list[tuple[str, int, str]]
            Short-form season, gameweek number, and repo-relative path.
        """
        tree = self.extractor.api_client.get_all_repo_files()
        found = []
        for entry in tree["tree"]:
            match = _MATCHSTATS_PATH_RE.match(entry["path"])
            if match:
                found.append(
                    (
                        season_long_to_short(match.group(1)),
                        int(match.group(2)),
                        entry["path"],
                    )
                )
        return found

    def available_seasons(self) -> list[str]:
        """Return the seasons FCI publishes match stats for.

        Returns
        -------
        list[str]
            Sorted short-form season strings.
        """
        return sorted({season for season, _, _ in self._matchstats_tree()})

    def matchstats_paths(self, season: str) -> dict[int, str]:
        """Return one repo path per gameweek for a season.

        Handles both layouts: 2024-25 stores files under
        ``playermatchstats/`` while later seasons use ``By Gameweek/``.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2025-26"``.

        Returns
        -------
        dict[int, str]
            Gameweek number to repo-relative path.
        """
        return {
            gw: path
            for found_season, gw, path in self._matchstats_tree()
            if found_season == season
        }

    def load_season(self, season: str) -> pl.DataFrame:
        """Download every gameweek of a season and shape it for storage.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2025-26"``.

        Returns
        -------
        pl.DataFrame
            Rows in ``PLAYER_MATCH_OPTA``'s exact column order and
            dtypes. Empty when the season has no gameweek files yet.
        """
        # season_short_to_long is not needed to build paths -- they come
        # from the tree -- but calling it validates the season string.
        season_short_to_long(season)
        frames = []
        for gw, path in sorted(self.matchstats_paths(season).items()):
            raw = self.extractor._read_csv(path)
            renamed = raw.rename({"player_id": "element"})
            unknown = PLAYER_MATCH_OPTA.unknown_columns(renamed)
            # ``season``, ``gw`` and ``competition`` are added below, so
            # they are never genuinely unknown.
            unknown = [
                column
                for column in unknown
                if column not in {"season", "gw", "competition"}
            ]
            if unknown:
                logger.warning(
                    "FCI %s GW%d carries %d column(s) not in the "
                    "player_match_opta schema; they will be dropped: %s",
                    season,
                    gw,
                    len(unknown),
                    ", ".join(unknown),
                )
            shaped = add_competition(
                renamed.with_columns(
                    pl.lit(season).alias("season"),
                    pl.lit(gw, dtype=pl.Int64).alias("gw"),
                )
            )
            frames.append(
                PLAYER_MATCH_OPTA.coerce(PLAYER_MATCH_OPTA.conform(shaped))
            )
        if not frames:
            logger.info("FCI has no match stats for season %s yet.", season)
            # PLAYER_MATCH_OPTA.conform(pl.DataFrame()) does not raise --
            # it silently broadcasts each typed-null literal to a single
            # row, producing a phantom (1, 67) frame instead of an empty
            # one. Build the empty frame explicitly instead.
            return pl.DataFrame(schema=PLAYER_MATCH_OPTA.schema).select(
                PLAYER_MATCH_OPTA.columns
            )
        return pl.concat(frames)

    def load(
        self,
        connection: "DuckDBPyConnection",
        current_season: str = CURRENT_SEASON,
    ) -> None:
        """Store every published season, skipping ones already loaded.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            An open connection.
        current_season : str, optional
            The season to upsert rather than treat as immutable.
            Defaults to ``CURRENT_SEASON``.
        """
        present = PLAYER_MATCH_OPTA.seasons_present(connection)
        for season in self.available_seasons():
            if season == current_season:
                PLAYER_MATCH_OPTA.upsert_current(
                    connection, self.load_season(season), season
                )
                continue
            if season in present:
                continue
            PLAYER_MATCH_OPTA.write_immutable(
                connection, self.load_season(season), season
            )
