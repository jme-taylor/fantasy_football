import logging
import math
import re
import time
from datetime import date, datetime
from typing import TYPE_CHECKING, Protocol

import polars as pl

from fantasy_football.storage.table import Table
from fantasy_football.storage.tables import (
    TM_MARKET_VALUE,
    TM_PLAYER,
    TM_PLAYER_SEASON,
    TM_TRANSFER,
)

if TYPE_CHECKING:
    import duckdb
    import pandas as pd

logger = logging.getLogger(__name__)

REQUEST_DELAY_SECONDS = 0.5
TRANSFERMARKT_LEAGUE = "England Premier League"

MONEY_REGEX = re.compile(r"(\d[\d.,]*)\s*(bn|m|k)?", re.IGNORECASE)
COST_MULTIPLIER_LOOKUP = {"bn": 1_000_000_000, "m": 1_000_000, "k": 1_000, None: 1}
DATE_FORMATS = ("%b %d, %Y", "%d.%m.%Y", "%Y-%m-%d")

class PlayerLinkSource(Protocol):
    """The slice of ``ScraperFC.Transfermarkt`` link collection calls."""

    def get_player_links(self, year: str, league: str) -> list[str]:
        """Return every player link in a league season."""
        ...


class TransfermarktClient(PlayerLinkSource, Protocol):
    """The slice of ``ScraperFC.Transfermarkt`` the player scrape calls."""

    def scrape_player(self, player_link: str) -> "pd.DataFrame":
        """Return a 1-row frame of one player's Transfermarkt page."""
        ...


def parse_money(value: str | None) -> int | None:
    """Parse a Transfermarkt money string into whole units of currency."""
    if value is None:
        return None
    match = MONEY_REGEX.search(value)
    if match is None:
        return None
    digits = match.group(1).replace(",", "")
    if digits.count(".") > 1:
        digits = digits.replace(".", "")
    try:
        amount = float(digits)
    except ValueError:
        return None
    suffix = match.group(2)
    multiplier = COST_MULTIPLIER_LOOKUP[suffix.lower() if suffix else None]
    return int(round(amount * multiplier))


def classify_fee(fee: str | None) -> str:
    """Classify a Transfermarkt fee string into a transfer type.

    Returns one of ``loan_end``, ``loan``, ``free``, ``transfer`` or
    ``unknown``. ``loan_end`` is tested before ``loan`` because "End of
    loan" contains both.
    """
    if fee is None:
        return "unknown"
    text = fee.strip().lower()
    if not text:
        return "unknown"
    if "end of loan" in text:
        return "loan_end"
    if "loan" in text:
        return "loan"
    if "free" in text:
        return "free"
    if parse_money(text) is not None:
        return "transfer"
    return "unknown"



def parse_tm_date(value: str | None) -> date | None:
    """Parse a Transfermarkt date string, returning None if it is not a date."""
    if value is None:
        return None
    text = value.strip()
    for fmt in DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def season_to_tm_year(season: str) -> str:
    """Convert a short-form season (``2016-17``) to Transfermarkt's ``16/17``."""
    start, end = season.split("-")
    return f"{start[-2:]}/{end}"


def player_id_from_link(player_link: str) -> str:
    """Return the Transfermarkt player id, the trailing segment of a player URL."""
    return player_link.rstrip("/").rsplit("/", 1)[-1]


def shape_player_season(season: str, player_links: list[str]) -> pl.DataFrame:
    """Shape one season's squad-list links into ``tm_player_season`` rows.

    Parameters
    ----------
    season : str
        Short-form season string, e.g. ``"2025-26"``.
    player_links : list[str]
        Transfermarkt player URLs collected for that season.

    Returns
    -------
    pl.DataFrame
        One row per distinct player, in ``TM_PLAYER_SEASON`` schema.
    """
    seen: dict[str, str] = {}
    for link in player_links:
        seen.setdefault(player_id_from_link(link), link)
    if not seen:
        return _empty(TM_PLAYER_SEASON)
    rows = [
        {"season": season, "tm_player_id": tm_player_id, "player_link": link}
        for tm_player_id, link in seen.items()
    ]
    return pl.DataFrame(rows).cast(TM_PLAYER_SEASON.schema, strict=False)


def _text(value: object) -> str | None:
    """Return a stripped string, or None for nulls and empty strings."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    text = str(value).strip()
    return text or None


def _number(value: object) -> float | None:
    """Return a float, or None for nulls and NaN.

    NaN is not a missing value in SQL, so letting one through would store a
    height that no ``IS NULL`` filter can find.
    """
    if value is None:
        return None
    try:
        number = float(value)  # ty: ignore[invalid-argument-type]
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _empty(table: Table) -> pl.DataFrame:
    """Return a zero-row frame carrying the table's schema."""
    return pl.DataFrame(schema=table.schema)


def _shape_transfers(
    tm_player_id: str, history: "pd.DataFrame | None"
) -> pl.DataFrame:
    """Shape a player's transfer history, newest transfer at seq 0."""
    if history is None or len(history) == 0:
        return _empty(TM_TRANSFER)
    rows = []
    for seq, (_, transfer) in enumerate(history.iterrows()):
        fee = _text(transfer.get("Fee"))
        market_value = _text(transfer.get("MV"))
        transfer_date = _text(transfer.get("Date"))
        rows.append(
            {
                "tm_player_id": tm_player_id,
                "transfer_seq": seq,
                "season": _text(transfer.get("Season")),
                "transfer_date": transfer_date,
                "transfer_date_parsed": parse_tm_date(transfer_date),
                "left_club": _text(transfer.get("Left")),
                "joined_club": _text(transfer.get("Joined")),
                "market_value": market_value,
                "market_value_eur": parse_money(market_value),
                "fee": fee,
                "fee_eur": parse_money(fee),
                "fee_type": classify_fee(fee),
            }
        )
    return pl.DataFrame(rows).cast(TM_TRANSFER.schema, strict=False)


def _shape_market_values(
    tm_player_id: str, history: "pd.DataFrame | None"
) -> pl.DataFrame:
    """Shape a player's market value history.

    The date is half the primary key, so a point whose date will not parse
    cannot be stored, and a date seen twice keeps only its first point.
    """
    if history is None or len(history) == 0:
        return _empty(TM_MARKET_VALUE)
    rows = []
    seen_dates: set[date] = set()
    for _, point in history.iterrows():
        raw_date = _text(point.get("date"))
        value_date = parse_tm_date(raw_date)
        if value_date in seen_dates:
            logger.warning(
                "Dropping repeated market value for %s on %s.",
                tm_player_id,
                value_date,
            )
            continue
        if value_date is None:
            logger.warning(
                "Dropping market value for %s: unparseable date %r.",
                tm_player_id,
                raw_date,
            )
            continue
        seen_dates.add(value_date)
        rows.append(
            {
                "tm_player_id": tm_player_id,
                "value_date": value_date,
                "value_date_raw": raw_date,
                "value_eur": _number(point.get("value")),
            }
        )
    if not rows:
        return _empty(TM_MARKET_VALUE)
    return pl.DataFrame(rows).cast(TM_MARKET_VALUE.schema, strict=False)


def shape_player(
    scraped: "pd.DataFrame",
    scraped_at: datetime,
    player_link: str | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Split one scraped player into its three stored frames.

    This is the pure seam between the network and storage: everything
    ``scrape_player`` returns is reshaped here, so the parsing is testable
    without an HTTP request.

    Parameters
    ----------
    scraped : pd.DataFrame
        The 1-row frame ``ScraperFC.Transfermarkt.scrape_player`` returns,
        with its nested market value and transfer history frames.
    scraped_at : datetime
        The instant the player was scraped.
    player_link : str | None, optional
        The URL the scrape came from. ``scrape_player`` does not return it,
        so it is stored only when the caller passes it back in.

    Returns
    -------
    tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]
        Frames for ``tm_player``, ``tm_transfer`` and ``tm_market_value``.

    Raises
    ------
    ValueError
        If the scrape carries no Transfermarkt player id, which is the key
        every one of the three frames hangs off.
    """
    row = scraped.iloc[0]
    # ScraperFC takes the id off the end of the URL without stripping a
    # trailing slash, so a link ending in one yields an empty id.
    tm_player_id = _text(row.get("ID"))
    if tm_player_id is None and player_link is not None:
        tm_player_id = _text(player_id_from_link(player_link))
    if tm_player_id is None:
        raise ValueError("Scraped player has no Transfermarkt id")

    citizenship = row.get("Citizenship")
    dob = _text(row.get("DOB"))
    since = _text(row.get("Since"))
    joined = _text(row.get("Joined"))
    contract_expiration = _text(row.get("Contract expiration"))
    value = _text(row.get("Value"))
    value_last_updated = _text(row.get("Value last updated"))

    player = pl.DataFrame(
        [
            {
                "tm_player_id": tm_player_id,
                "name": _text(row.get("Name")),
                "player_link": player_link,
                "dob": dob,
                "dob_date": parse_tm_date(dob),
                "height_m": _number(row.get("Height (m)")),
                "nationality": _text(row.get("Nationality")),
                "citizenship": (
                    "|".join(sorted(str(c) for c in citizenship))
                    if isinstance(citizenship, (list, tuple))
                    and len(citizenship)
                    else None
                ),
                "position": _text(row.get("Position")),
                "team": _text(row.get("Team")),
                "last_club": _text(row.get("Last club")),
                "since": since,
                "since_date": parse_tm_date(since),
                "joined": joined,
                "joined_date": parse_tm_date(joined),
                "contract_expiration": contract_expiration,
                "contract_expiration_date": parse_tm_date(contract_expiration),
                "value": value,
                "value_eur": parse_money(value),
                "value_last_updated": value_last_updated,
                "value_last_updated_date": parse_tm_date(value_last_updated),
                "scraped_at": scraped_at,
            }
        ]
    ).cast(TM_PLAYER.schema, strict=False)

    transfers = _shape_transfers(tm_player_id, row.get("Transfer history"))
    market_values = _shape_market_values(
        tm_player_id, row.get("Market value history")
    )
    return player, transfers, market_values


def store_player(
    connection: "duckdb.DuckDBPyConnection",
    player: pl.DataFrame,
    transfers: pl.DataFrame,
    market_values: pl.DataFrame,
) -> None:
    """Write one player's three frames, replacing anything already stored.

    Writing per player rather than per batch is what makes the scrape
    resumable to the exact player: two HTTP round-trips dominate the cost,
    so three small statements alongside them are free.

    ``tm_player`` is written last because it is what the resume anti-join
    reads. Written first, a failure on either history table would leave the
    player looking scraped and no run would ever retry them.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    player, transfers, market_values : pl.DataFrame
        The three frames ``shape_player`` returned.
    """
    tm_player_id = player["tm_player_id"][0]
    partition = {"tm_player_id": tm_player_id}
    TM_TRANSFER.replace_partition(connection, transfers, partition)
    TM_MARKET_VALUE.replace_partition(connection, market_values, partition)
    TM_PLAYER.replace_partition(connection, player, partition)


def pending_player_links(
    connection: "duckdb.DuckDBPyConnection",
    refresh: bool = False,
    limit: int | None = None,
) -> list[str]:
    """Return the player links still to scrape, newest season first.

    The anti-join against ``tm_player`` is the resume mechanism: a run that
    dies partway is continued by re-running it, and a player who failed is
    retried on the next run without any checkpoint bookkeeping.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    refresh : bool, optional
        When True, return every link regardless of what is already stored.
    limit : int | None, optional
        Return at most this many links.

    Returns
    -------
    list[str]
        One link per distinct player.
    """
    query = """
        SELECT ps.player_link
        FROM (
            SELECT tm_player_id, max(season) AS season, min(player_link) AS player_link
            FROM tm_player_season
            GROUP BY tm_player_id
        ) AS ps
    """
    if not refresh:
        query += """
        LEFT JOIN tm_player AS p USING (tm_player_id)
        WHERE p.tm_player_id IS NULL
        """
    query += " ORDER BY ps.season DESC, ps.tm_player_id"
    if limit is not None:
        query += f" LIMIT {int(limit)}"
    return [row[0] for row in connection.execute(query).fetchall()]


def collect_player_links(
    connection: "duckdb.DuckDBPyConnection",
    seasons: list[str],
    transfermarkt: PlayerLinkSource,
) -> None:
    """Refresh ``tm_player_season`` from each season's squad lists.

    A season is only replaced when the scrape actually returned players.
    ScraperFC warns and returns an empty list when Transfermarkt's clubs
    table is missing -- a block or a layout change -- and replacing a
    season with that would delete the very link list the scrape resumes
    from. A season that raises is logged and skipped for the same reason.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    seasons : list[str]
        Short-form seasons to collect links for.
    transfermarkt : PlayerLinkSource
        A ScraperFC Transfermarkt client.
    """
    for season in seasons:
        try:
            links = transfermarkt.get_player_links(
                year=season_to_tm_year(season), league=TRANSFERMARKT_LEAGUE
            )
        except Exception:
            logger.exception(
                "Failed to collect links for %s; skipping.", season
            )
            continue
        frame = shape_player_season(season, list(links))
        if frame.is_empty():
            logger.warning(
                "Transfermarkt returned no players for %s; keeping the "
                "stored links rather than replacing them with nothing.",
                season,
            )
            continue
        TM_PLAYER_SEASON.upsert_current(connection, frame, season)
        logger.info("Collected %d player links for %s.", frame.height, season)


def scrape_one(
    player_link: str, transfermarkt: TransfermarktClient, scraped_at: datetime
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Scrape one player page and shape it into its three frames."""
    scraped = transfermarkt.scrape_player(player_link)
    return shape_player(scraped, scraped_at, player_link=player_link)


def scrape_players(
    connection: "duckdb.DuckDBPyConnection",
    transfermarkt: TransfermarktClient,
    refresh: bool = False,
    limit: int | None = None,
    delay: float = REQUEST_DELAY_SECONDS,
) -> None:
    """Scrape every player still missing from ``tm_player``.

    A player that fails is logged and skipped rather than stopping the run:
    the next run's anti-join picks it up again, so resumability is also the
    retry mechanism.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    transfermarkt : TransfermarktClient
        A ScraperFC Transfermarkt client.
    refresh : bool, optional
        Re-scrape players already stored.
    limit : int | None, optional
        Stop after this many players.
    delay : float, optional
        Seconds to wait between players.
    """
    links = pending_player_links(connection, refresh=refresh, limit=limit)
    logger.info("Scraping %d Transfermarkt players.", len(links))
    failures = 0
    for index, link in enumerate(links):
        if index:
            time.sleep(delay)
        try:
            frames = scrape_one(link, transfermarkt, datetime.now())
            store_player(connection, *frames)
        except Exception:
            failures += 1
            logger.exception("Failed to scrape %s; skipping.", link)
            continue
        if (index + 1) % 50 == 0:
            logger.info("Scraped %d/%d players.", index + 1, len(links))
    logger.info(
        "Finished: %d scraped, %d failed.", len(links) - failures, failures
    )
