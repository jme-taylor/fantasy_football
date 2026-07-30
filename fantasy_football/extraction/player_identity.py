"""Build ``player_season`` rows -- the cross-season player identity spine.

FPL reassigns ``element`` ids every season, so ``(season, element)`` cannot join
a player to their own past. FPL's ``code`` is stable and global, and every
upstream source already carries it under a different name: Vaastav calls it
``code`` (with ``id`` for the element), FCI calls it ``player_code`` (with
``player_id``). This module normalises both into one shape.
"""

import logging

import polars as pl

from fantasy_football.extraction.fci import FCI_POSITION_TO_VAASTAV
from fantasy_football.storage.database import (
    PLAYER_SEASON_COLUMNS,
    coerce_player_season,
)

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
