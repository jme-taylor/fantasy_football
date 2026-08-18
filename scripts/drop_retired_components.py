"""Delete component rows for components no longer declared.

    uv run python scripts/drop_retired_components.py

A one-off migration, not a pipeline step. ``compose`` refuses to run
while a position carries a component it no longer declares -- deliberately,
since a stale row would otherwise be summed on top of the components that
replaced it. Decomposing a position leaves exactly those rows behind, so
they have to go before the next pipeline run -- most recently GK, whose
``total`` rows predate its split into appearance, saves and conceding.

Safe to run more than once: it deletes what is stale now and reports the
count, so a second run deletes nothing.
"""

import logging

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.logging_config import configure_logging
from fantasy_football.modelling.components import POSITION_COMPONENTS
from fantasy_football.storage.database import get_connection
from fantasy_football.storage.tables import POINTS_COMPONENT

logger = logging.getLogger(__name__)


def _count(row: tuple | None) -> int:
    """Return the row count DuckDB reports for a DELETE."""
    return int(row[0]) if row else 0


def main() -> None:
    """Delete every stored component its position no longer declares."""
    configure_logging()
    connection = get_connection(DATABASE_PATH)
    try:
        for position, components in POSITION_COMPONENTS.items():
            declared = ", ".join(f"'{component}'" for component in components)
            deleted = connection.execute(
                f"DELETE FROM {POINTS_COMPONENT.name} "
                f"WHERE position = ? AND component NOT IN ({declared})",
                [position],
            ).fetchone()
            logger.info("Deleted %d rows for %s", _count(deleted), position)
        # A position no longer in the table at all -- there is none
        # today, but a row for one would fail compose the same way.
        known = ", ".join(f"'{name}'" for name in POSITION_COMPONENTS)
        deleted = connection.execute(
            f"DELETE FROM {POINTS_COMPONENT.name} "
            f"WHERE position NOT IN ({known})"
        ).fetchone()
        logger.info("Deleted %d rows for unknown positions", _count(deleted))
    finally:
        connection.close()


if __name__ == "__main__":
    main()
