"""One-off migration: add ``status`` to ``player_snapshot``.

``CREATE TABLE IF NOT EXISTS`` cannot add a column, so an existing
database file fails ``check_schema_drift`` once the spec declares
``status``. Adding it in place is cheaper than a full rebuild; the
column is null for every capture taken before this ran, which
``latest_snapshot`` treats as "not departed".

Run once, with nothing else holding the database lock:

    uv run python scripts/add_snapshot_status.py
"""

import duckdb

from fantasy_football.constants import CURRENT_SEASON, DATABASE_PATH
from fantasy_football.extraction.snapshot import load_player_snapshot


def main() -> None:
    """Add the column, then capture a snapshot that populates it."""
    connection = duckdb.connect(str(DATABASE_PATH))
    try:
        connection.execute(
            "ALTER TABLE player_snapshot ADD COLUMN IF NOT EXISTS "
            "status VARCHAR"
        )
        load_player_snapshot(CURRENT_SEASON, connection)
    finally:
        connection.close()


if __name__ == "__main__":
    main()
