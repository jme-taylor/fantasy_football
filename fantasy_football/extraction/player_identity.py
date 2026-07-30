"""Build ``player_season`` rows -- the cross-season player identity spine.

FPL reassigns ``element`` ids every season, so ``(season, element)`` cannot join
a player to their own past. FPL's ``code`` is stable and global, and every
upstream source already carries it under a different name: Vaastav calls it
``code`` (with ``id`` for the element), FCI calls it ``player_code`` (with
``player_id``). This module normalises both into one shape.
"""

import logging
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.constants import (
    CURRENT_SEASON,
    EARLIEST_IDENTITY_SEASON,
    FPLCACHE_FIRST_SEASON,
)
from fantasy_football.extraction.extractor import DataExtractor
from fantasy_football.extraction.fci import (
    FCI_POSITION_TO_VAASTAV,
    FciExtractor,
)
from fantasy_football.extraction.fplcache import FplCacheExtractor
from fantasy_football.extraction.seasons import (
    DataSource,
    season_short_to_long,
    seasons_in_range,
    source_for_season,
)
from fantasy_football.storage.database import (
    PLAYER_SEASON_COLUMNS,
    coerce_player_season,
    player_season_seasons_present,
    upsert_current_player_season,
    write_immutable_player_season,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# FPL's element_type integer -> the canonical position label used everywhere
# else in the pipeline. Vaastav's players_raw.csv carries the integer; FCI
# carries a long-form label handled by FCI_POSITION_TO_VAASTAV.
ELEMENT_TYPE_TO_POSITION: dict[int, str] = {
    1: "GK",
    2: "DEF",
    3: "MID",
    4: "FWD",
}


def _check_identity_integrity(frame: pl.DataFrame, season: str) -> None:
    """Raise if a season maps an element or a code more than once.

    Either condition means the source file is malformed. Failing loudly beats
    silently corrupting every history feature that joins through the mapping.

    Parameters
    ----------
    frame : pl.DataFrame
        A shaped player-season frame with ``element`` and ``player_code``.
    season : str
        Short-form season string, used in the error message.

    Raises
    ------
    ValueError
        If any ``element`` or any non-null ``player_code`` occurs twice.
    """
    if frame["element"].is_duplicated().any():
        duplicates = (
            frame.filter(pl.col("element").is_duplicated())["element"]
            .unique()
            .to_list()
        )
        raise ValueError(
            f"Season {season} has duplicate element ids: {sorted(duplicates)}"
        )
    codes = frame.filter(pl.col("player_code").is_not_null())
    if codes["player_code"].is_duplicated().any():
        duplicates = (
            codes.filter(pl.col("player_code").is_duplicated())["player_code"]
            .unique()
            .to_list()
        )
        raise ValueError(
            f"Season {season} has duplicate player_code values: "
            f"{sorted(duplicates)}"
        )


def _finalise(frame: pl.DataFrame, season: str) -> pl.DataFrame:
    """Add any absent optional columns, coerce, and check integrity.

    Older Vaastav files predate ``birth_date``, ``region`` and
    ``team_join_date``, and FCI never supplies ``birth_date``. Missing columns
    become explicit nulls so the frame always matches the table.

    Parameters
    ----------
    frame : pl.DataFrame
        A partially shaped frame carrying at least ``season``, ``element`` and
        ``player_code``.
    season : str
        Short-form season string.

    Returns
    -------
    pl.DataFrame
        A frame with exactly ``PLAYER_SEASON_COLUMNS``.
    """
    missing = [
        column
        for column in PLAYER_SEASON_COLUMNS
        if column not in frame.columns
    ]
    if missing:
        frame = frame.with_columns(
            [pl.lit(None).alias(column) for column in missing]
        )
    shaped = coerce_player_season(frame)
    _check_identity_integrity(shaped, season)
    return shaped


def build_player_season_from_vaastav(
    players_raw: pl.DataFrame, season: str
) -> pl.DataFrame:
    """Shape a Vaastav ``players_raw.csv`` frame into player-season rows.

    Parameters
    ----------
    players_raw : pl.DataFrame
        A parsed ``data/{season}/players_raw.csv``. Must contain ``id``,
        ``code``, ``web_name``, ``first_name``, ``second_name``,
        ``element_type`` and ``team_code``. ``birth_date`` is used when present
        (2024-25 onwards) and nulled otherwise.
    season : str
        Short-form season string, e.g. ``"2023-24"``.

    Returns
    -------
    pl.DataFrame
        One row per ``(season, element)`` with ``PLAYER_SEASON_COLUMNS``.

    Raises
    ------
    ValueError
        If the season maps an element or a player_code more than once.
    """
    frame = players_raw.rename({"id": "element", "code": "player_code"})
    frame = frame.with_columns(
        pl.lit(season).alias("season"),
        pl.col("element_type")
        .cast(pl.Int64, strict=False)
        .replace(ELEMENT_TYPE_TO_POSITION, default=None)
        .alias("position"),
    )
    if "birth_date" in frame.columns:
        frame = frame.with_columns(
            pl.col("birth_date")
            .cast(pl.Utf8)
            .str.to_date(format="%Y-%m-%d", strict=False)
        )
    return _finalise(frame, season)


def build_player_season_from_fci(
    players: pl.DataFrame, season: str
) -> pl.DataFrame:
    """Shape an FCI ``players.csv`` frame into player-season rows.

    FCI supplies no ``birth_date``; it is nulled here and filled from fplcache
    by the enrichment pass, or propagated from another season at load time.

    Parameters
    ----------
    players : pl.DataFrame
        A parsed ``data/{long_season}/players.csv``. Must contain
        ``player_id``, ``player_code``, ``web_name``, ``first_name``,
        ``second_name``, ``position`` and ``team_code``.
    season : str
        Short-form season string, e.g. ``"2025-26"``.

    Returns
    -------
    pl.DataFrame
        One row per ``(season, element)`` with ``PLAYER_SEASON_COLUMNS``.

    Raises
    ------
    ValueError
        If the season maps an element or a player_code more than once.
    """
    frame = players.rename({"player_id": "element"})
    frame = frame.with_columns(
        pl.lit(season).alias("season"),
        pl.col("position")
        .replace(FCI_POSITION_TO_VAASTAV, default=None)
        .alias("position"),
    )
    return _finalise(frame, season)


def _enrich_with_bio(frame: pl.DataFrame, bio: pl.DataFrame) -> pl.DataFrame:
    """Overlay fplcache bio columns onto a shaped player-season frame.

    The source builders always emit ``birth_date``, ``region`` and
    ``team_join_date`` (null when the source has no such column), so the join
    coalesces the incoming values over the existing ones rather than adding
    columns.

    Parameters
    ----------
    frame : pl.DataFrame
        A shaped player-season frame with ``PLAYER_SEASON_COLUMNS``.
    bio : pl.DataFrame
        Rows from ``FplCacheExtractor.build_player_bio``.

    Returns
    -------
    pl.DataFrame
        ``frame`` with bio columns filled where fplcache had a value.
    """
    if bio.is_empty():
        return frame
    bio_columns = ["birth_date", "region", "team_join_date"]
    joined = frame.join(
        bio.rename({column: f"{column}_bio" for column in bio_columns}),
        on="element",
        how="left",
        coalesce=True,
    )
    return joined.with_columns(
        [
            pl.coalesce([f"{column}_bio", column]).alias(column)
            for column in bio_columns
        ]
    ).select(PLAYER_SEASON_COLUMNS)


def load_player_identity_data(
    connection: "DuckDBPyConnection",
    current_season: str = CURRENT_SEASON,
    vaastav: DataExtractor | None = None,
    fci: FciExtractor | None = None,
    fplcache: FplCacheExtractor | None = None,
) -> None:
    """Populate ``player_season`` for every season up to ``current_season``.

    Completed seasons are fetched once and skipped thereafter; the current
    season is upserted each run so newly-registered players appear. Seasons
    from ``FPLCACHE_FIRST_SEASON`` onwards are enriched with bio fields no
    other source carries.

    A season whose source file cannot be fetched is logged at WARNING and
    skipped. A gap in the middle of history must not stop the current season
    from loading, and a missing season simply means no history features for the
    players who only appear in it.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        Open connection to the database.
    current_season : str, optional
        Short-form current season. Defaults to ``CURRENT_SEASON``.
    vaastav : DataExtractor | None, optional
        Vaastav extractor. Defaults to a new ``DataExtractor``.
    fci : FciExtractor | None, optional
        FCI extractor. Defaults to a new ``FciExtractor``.
    fplcache : FplCacheExtractor | None, optional
        fplcache extractor. Defaults to a new ``FplCacheExtractor``.
    """
    vaastav = vaastav or DataExtractor()
    fci = fci or FciExtractor()
    fplcache = fplcache or FplCacheExtractor()
    present = player_season_seasons_present(connection)

    for season in seasons_in_range(EARLIEST_IDENTITY_SEASON, current_season):
        if season != current_season and season in present:
            logger.info(
                "Player-season season %s already present; skipping fetch.",
                season,
            )
            continue
        try:
            if source_for_season(season) == DataSource.VAASTAV:
                frame = build_player_season_from_vaastav(
                    vaastav.read_players_raw(season), season
                )
            else:
                frame = build_player_season_from_fci(
                    fci.read_players(season_short_to_long(season)), season
                )
        except Exception:
            logger.warning(
                "Could not fetch player identity for season %s; skipping.",
                season,
                exc_info=True,
            )
            continue

        if season >= FPLCACHE_FIRST_SEASON:
            try:
                frame = _enrich_with_bio(
                    frame, fplcache.build_player_bio(season)
                )
            except Exception:
                logger.warning(
                    "Could not fetch fplcache bio for season %s; continuing "
                    "without it.",
                    season,
                    exc_info=True,
                )

        if season == current_season:
            upsert_current_player_season(connection, frame, season)
        else:
            write_immutable_player_season(connection, frame, season)
