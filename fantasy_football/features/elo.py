import logging
from datetime import datetime
from pathlib import Path

import pandas as pd
import polars as pl
from ScraperFC import ClubElo

from fantasy_football.constants import (
    CLUBELO_SCRAPE_NAMES,
    CLUBELO_TO_FPL,
    ELO_CACHE_TTL_HOURS,
    ELO_HISTORY_START,
    TRANSFORMED_DATA_FOLDER,
)

logger = logging.getLogger(__name__)

_OUTPUT_FILE = "team_elo.csv"


def normalize_elo_frame(raw: pd.DataFrame) -> pl.DataFrame:
    """Normalize a ClubElo-shaped pandas DataFrame to our polars schema.

    Maps `Club` -> `team` using ``CLUBELO_TO_FPL``, emitting one row per FPL
    alias so a club renamed between seasons resolves under either name.
    Unknown clubs are dropped with a logged warning.

    Parameters
    ----------
    raw: pd.DataFrame
        The raw ClubElo DataFrame to normalize.

    Returns
    -------
    pl.DataFrame
        The normalized ClubElo DataFrame.
    """
    if raw.empty:
        return pl.DataFrame(
            schema={
                "team": pl.Utf8,
                "elo": pl.Float64,
                "from_date": pl.Date,
                "to_date": pl.Date,
            }
        )
    df = pl.from_pandas(raw[["Club", "Elo", "From", "To"]]).rename(
        {"Club": "club", "Elo": "elo", "From": "from_date", "To": "to_date"}
    )
    known = set(CLUBELO_TO_FPL)
    unknown = sorted(set(df["club"].unique().to_list()) - known)
    for name in unknown:
        logger.warning("Unknown ClubElo team %r — dropping rows", name)
    df = df.filter(pl.col("club").is_in(list(known)))
    alias_frame = pl.DataFrame(
        {
            "club": list(CLUBELO_TO_FPL),
            "team": [CLUBELO_TO_FPL[club] for club in CLUBELO_TO_FPL],
        },
        schema={"club": pl.Utf8, "team": pl.List(pl.Utf8)},
    ).explode("team")
    return (
        df.with_columns(
            pl.col("from_date").str.strptime(pl.Date, "%Y-%m-%d"),
            pl.col("to_date").str.strptime(pl.Date, "%Y-%m-%d"),
        )
        .join(alias_frame, on="club", how="inner")
        .filter(pl.col("to_date") >= ELO_HISTORY_START)
        .select("team", "elo", "from_date", "to_date")
    )


def _cache_is_fresh(path: Path) -> bool:
    if not path.exists():
        return False
    age_hours = (datetime.now().timestamp() - path.stat().st_mtime) / 3600
    return age_hours < ELO_CACHE_TTL_HOURS


# TODO(JT): Try and remove pandas dependecy here
def build_team_elo(*, force: bool = False) -> pl.DataFrame:
    """Scrape ClubElo for every team in ``CLUBELO_TO_FPL`` and write CSV.

    Reads cached CSV if it exists and is fresher than ``ELO_CACHE_TTL_HOURS``
    (unless ``force=True``). On scrape failure, falls back to the existing
    cache if one exists; otherwise re-raises.

    Parameters
    ----------
    force: bool
        If True, force a re-scrape of ClubElo.

    Returns
    -------
    pl.DataFrame
        The Team ELO DataFrame.
    """
    TRANSFORMED_DATA_FOLDER.mkdir(exist_ok=True, parents=True)
    cache_path = TRANSFORMED_DATA_FOLDER.joinpath(_OUTPUT_FILE)

    if not force and _cache_is_fresh(cache_path):
        logger.info("Using cached ELO data at %s", cache_path)
        return pl.read_csv(cache_path, try_parse_dates=True)

    client = ClubElo()
    frames: list[pd.DataFrame] = []
    try:
        for slug in CLUBELO_SCRAPE_NAMES:
            frames.append(client.scrape_team(slug))
    except Exception as exc:
        if cache_path.exists():
            logger.warning(
                "ELO scrape failed (%s); falling back to cache at %s",
                exc,
                cache_path,
            )
            return pl.read_csv(cache_path, try_parse_dates=True)
        raise

    combined = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=pd.Index(
                ["Rank", "Club", "Country", "Level", "Elo", "From", "To"]
            )
        )
    )
    df = normalize_elo_frame(combined)
    df.write_csv(cache_path)
    logger.info("Wrote %d ELO rows to %s", df.height, cache_path)
    return df
