"""Load Vaastav's full per-fixture player stats into ``player_match_fpl``.

The per-season ``data/<season>/gws/merged_gw.csv`` file is a superset of
the per-player ``players/*/gw.csv`` files -- it adds ``name``,
``position``, ``team``, ``GW`` and ``xP`` -- so this loader makes one
request per season rather than roughly seven hundred.

Unlike ``extractor.load_immutable_player_match_seasons``, which narrows
to eight columns, this keeps everything the source publishes. Columns
differ by season; ``Table.conform`` fills the gaps with typed nulls and
``storage/coverage.py`` records which gaps are which.
"""

import logging
import re
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import CURRENT_SEASON
from fantasy_football.extraction.extractor import (
    DataExtractor,
    _drop_duplicate_rows,
)
from fantasy_football.storage.tables import PLAYER_MATCH_FPL

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# Seasons whose merged_gw.csv contains latin-1 bytes and cannot be read
# as strict UTF-8.
LOSSY_ENCODING_SEASONS: frozenset[str] = frozenset(
    {"2016-17", "2017-18", "2018-19"}
)

_MERGED_GW_PATH_RE = re.compile(r"^data/(\d{4}-\d{2})/gws/merged_gw\.csv$")


def shape_merged_gw(frame: pl.DataFrame, season: str) -> pl.DataFrame:
    """Rename the source's gameweek column, stamp the season, dedupe.

    Rows repeated verbatim in the source are dropped -- 2025-26 carries
    ten of them -- because they would otherwise violate the table's
    ``(season, gw, element, fixture)`` primary key. Both legs of a
    genuine double gameweek differ by ``fixture`` and survive.

    Parameters
    ----------
    frame : pl.DataFrame
        A merged_gw.csv as read from the source.
    season : str
        Short-form season string, e.g. ``"2016-17"``.

    Returns
    -------
    pl.DataFrame
        The frame with ``gw`` and ``season`` columns, deduplicated.
    """
    shaped = frame.rename({"GW": "gw"}).with_columns(
        pl.lit(season).alias("season")
    )
    return _drop_duplicate_rows(shaped, f"{season} merged_gw.csv")


class VaastavMatchLoader:
    """Download Vaastav per-fixture stats and store them by season."""

    def __init__(self, extractor: DataExtractor | None = None) -> None:
        """Initialise the loader.

        Parameters
        ----------
        extractor : DataExtractor | None, optional
            Vaastav extractor. Defaults to a new ``DataExtractor``.
        """
        self.extractor = extractor or DataExtractor()

    def available_seasons(self) -> list[str]:
        """Return the seasons the repo publishes a merged_gw.csv for.

        Discovered from the repo tree rather than hardcoded, so a newly
        published season loads itself and an absent one -- 2026-27, as
        of writing -- is simply not returned.

        Returns
        -------
        list[str]
            Sorted short-form season strings.
        """
        tree = self.extractor.api_client.get_all_repo_files()
        seasons = set()
        for entry in tree["tree"]:
            match = _MERGED_GW_PATH_RE.match(entry["path"])
            if match:
                seasons.add(match.group(1))
        return sorted(seasons)

    def load_season(self, season: str) -> pl.DataFrame:
        """Download one season and reduce it to the table's schema.

        Parameters
        ----------
        season : str
            Short-form season string, e.g. ``"2016-17"``.

        Returns
        -------
        pl.DataFrame
            Rows in ``PLAYER_MATCH_FPL``'s exact column order and dtypes.
        """
        encoding = "utf8-lossy" if season in LOSSY_ENCODING_SEASONS else "utf8"
        raw = self.extractor._read_csv(
            f"data/{season}/gws/merged_gw.csv", encoding=encoding
        )
        shaped = shape_merged_gw(raw, season)
        unknown = PLAYER_MATCH_FPL.unknown_columns(shaped)
        if unknown:
            logger.warning(
                "Vaastav season %s carries %d column(s) not in the "
                "player_match_fpl schema; they will be dropped: %s",
                season,
                len(unknown),
                ", ".join(unknown),
            )
        return PLAYER_MATCH_FPL.coerce(PLAYER_MATCH_FPL.conform(shaped))

    def load(
        self,
        connection: "DuckDBPyConnection",
        current_season: str = CURRENT_SEASON,
    ) -> None:
        """Store every published season, skipping ones already loaded.

        Completed seasons are written immutably, so a normal run touches
        the network only for seasons not yet stored. The current season,
        when the source carries it at all, is upserted instead.

        Parameters
        ----------
        connection : duckdb.DuckDBPyConnection
            An open connection.
        current_season : str, optional
            The season to upsert rather than treat as immutable.
            Defaults to ``CURRENT_SEASON``.
        """
        present = PLAYER_MATCH_FPL.seasons_present(connection)
        for season in self.available_seasons():
            if season == current_season:
                frame = self.load_season(season)
                if frame.is_empty():
                    # PLAYER_MATCH_FPL.upsert_current deletes the
                    # season's rows before inserting, so upserting an
                    # empty frame would wipe whatever is already stored.
                    # Leave the database untouched instead (see
                    # fci.py's build_current_season_merged_gw for the
                    # same guard).
                    logger.warning(
                        "Vaastav has no rows for current season %s; "
                        "leaving the stored season untouched.",
                        season,
                    )
                    continue
                PLAYER_MATCH_FPL.upsert_current(connection, frame, season)
                continue
            if season in present:
                continue
            PLAYER_MATCH_FPL.write_immutable(
                connection, self.load_season(season), season
            )
