"""Scrape Transfermarkt player data into the raw ``tm_*`` tables.

    uv run python scripts/scrape_transfermarkt.py

Two phases. The first collects each season's squad-list links into
``tm_player_season``; the second scrapes every player in that list not
already in ``tm_player``. That anti-join makes the run resumable -- kill
it and re-run and it continues where it stopped -- so a multi-hour scrape
never has to start over.

Either phase can be run alone: ``--links-only`` stops after collection,
``--scrape-only`` skips it, so a resumed scrape is never blocked by a
season Transfermarkt has not published yet.

The run finishes by rebuilding ``tm_player_map``, the person-level bridge
from FPL to Transfermarkt, and writing a candidate report for whatever it
could not match. ``--map-only`` rebuilds just that, which is what to run
after editing the override CSV.

Nothing in ``main.py`` populates these tables, so ``main(rebuild=True)``
drops them and only a re-scrape brings them back.
"""

import argparse
import logging

import ScraperFC as sfc

from fantasy_football.constants import (
    CURRENT_SEASON,
    DATABASE_PATH,
    EARLIEST_IDENTITY_SEASON,
)
from fantasy_football.extraction.seasons import seasons_in_range
from fantasy_football.extraction.tm_player_map import (
    TransferMarktPlayerMap,
)
from fantasy_football.extraction.transfermarkt import (
    REQUEST_DELAY_SECONDS,
    collect_player_links,
    scrape_players,
)
from fantasy_football.logging_config import configure_logging
from fantasy_football.storage.database import get_connection

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seasons",
        nargs="+",
        help="Short-form seasons to collect links for. Defaults to every "
        "season with player-week data.",
    )
    parser.add_argument(
        "--links-only",
        action="store_true",
        help="Collect season squad links and stop, without scraping players.",
    )
    parser.add_argument(
        "--scrape-only",
        action="store_true",
        help="Scrape players from the stored links without re-collecting "
        "them, so a resumed run cannot be blocked by the link phase.",
    )
    parser.add_argument(
        "--map-only",
        action="store_true",
        help="Rebuild tm_player_map from the stored scrape and the "
        "mapping CSVs, without scraping anything.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-scrape players already stored, rather than resuming.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Stop after this many players.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY_SECONDS,
        help="Seconds to wait between players.",
    )
    return parser.parse_args()


def main() -> None:
    """Collect season links, scrape what is missing, then rebuild the map."""
    configure_logging()
    args = parse_args()
    seasons = args.seasons or seasons_in_range(
        EARLIEST_IDENTITY_SEASON, CURRENT_SEASON
    )
    transfermarkt = sfc.Transfermarkt()
    connection = get_connection(DATABASE_PATH)
    try:
        if args.map_only:
            TransferMarktPlayerMap(connection).refresh_player_map()
            return
        if not args.scrape_only:
            collect_player_links(connection, seasons, transfermarkt)
        if args.links_only:
            return
        scrape_players(
            connection,
            transfermarkt,
            refresh=args.refresh,
            limit=args.limit,
            delay=args.delay,
        )
        TransferMarktPlayerMap(connection).refresh_player_map()
    finally:
        connection.close()


if __name__ == "__main__":
    main()
