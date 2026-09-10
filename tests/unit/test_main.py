from datetime import date, datetime, timedelta

import duckdb
import polars as pl
import pytest

from fantasy_football.storage.tables import (
    PLAYER_SEASON,
    PLAYER_WEEK,
    TM_MARKET_VALUE,
    TM_PLAYER_MAP,
)
from main import TM_MAX_VALUE_AGE_DAYS, check_transfermarkt_freshness

SEASON = "2024-25"


def _seed_valuations(
    connection: duckdb.DuckDBPyConnection, age_days: int
) -> None:
    TM_MARKET_VALUE.append(
        connection,
        pl.DataFrame(
            {
                "tm_player_id": ["a"],
                "value_date": [date.today() - timedelta(days=age_days)],
                "value_date_raw": ["x"],
                "value_eur": [1_000_000],
            }
        ),
    )


def _seed_appearances(
    connection: duckdb.DuckDBPyConnection, elements: list[int]
) -> None:
    n = len(elements)
    PLAYER_WEEK.append(
        connection,
        pl.DataFrame(
            {
                "season": [SEASON] * n,
                "gw": [1] * n,
                "element": elements,
                "name": [f"P{e}" for e in elements],
                "position": ["MID"] * n,
                "team": ["Arsenal"] * n,
                "bonus": [0] * n,
                "minutes": [90] * n,
                "round": [1] * n,
                "total_points": [2] * n,
                "value": [50] * n,
            }
        ),
    )
    PLAYER_SEASON.append(
        connection,
        PLAYER_SEASON.conform(
            pl.DataFrame(
                {
                    "season": [SEASON] * n,
                    "element": elements,
                    "player_code": [e * 100 for e in elements],
                    "birth_date": [date(1995, 1, 1)] * n,
                    "team_join_date": [date(2020, 1, 1)] * n,
                }
            )
        ),
    )


def _seed_map(
    connection: duckdb.DuckDBPyConnection, player_codes: list[int]
) -> None:
    n = len(player_codes)
    TM_PLAYER_MAP.append(
        connection,
        pl.DataFrame(
            {
                "player_code": player_codes,
                "tm_player_id": [f"tm{c}" for c in player_codes],
                "match_rung": ["exact"] * n,
                "match_score": [1.0] * n,
                "fpl_name": [f"P{c}" for c in player_codes],
                "tm_name": [f"P{c}" for c in player_codes],
            }
        ),
    )


def test_empty_market_value_table_fails(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """No valuations at all means the scrape has never run."""
    with pytest.raises(RuntimeError, match="is empty"):
        check_transfermarkt_freshness(db, SEASON)


def test_stale_valuations_fail(db: duckdb.DuckDBPyConnection) -> None:
    """A scrape older than the limit is a loud failure, not a warning."""
    _seed_valuations(db, TM_MAX_VALUE_AGE_DAYS + 1)

    with pytest.raises(RuntimeError, match="days old"):
        check_transfermarkt_freshness(db, SEASON)


def test_thin_map_coverage_fails(db: duckdb.DuckDBPyConnection) -> None:
    """Fresh valuations do not excuse players the map never matched."""
    _seed_valuations(db, 1)
    _seed_appearances(db, [1, 2, 3, 4])
    _seed_map(db, [100])

    with pytest.raises(RuntimeError, match="map to Transfermarkt"):
        check_transfermarkt_freshness(db, SEASON)


def test_fresh_and_covered_passes(db: duckdb.DuckDBPyConnection) -> None:
    """Recent valuations plus a fully-mapped squad clears both checks."""
    _seed_valuations(db, 1)
    _seed_appearances(db, [1, 2])
    _seed_map(db, [100, 200])

    check_transfermarkt_freshness(db, SEASON)


def test_no_appearances_yet_checks_freshness_only(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Before a ball is kicked, coverage is unmeasurable rather than zero."""
    _seed_valuations(db, 1)

    check_transfermarkt_freshness(db, SEASON)


def test_unplayed_rows_do_not_count_towards_coverage(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A wide FPL squad of non-appearers cannot fail the coverage check.

    FPL squads carry youth players Transfermarkt never lists. Only players
    who have actually taken the pitch are held against the map.
    """
    _seed_valuations(db, 1)
    _seed_appearances(db, [1])
    _seed_map(db, [100])
    PLAYER_WEEK.append(
        db,
        pl.DataFrame(
            {
                "season": [SEASON] * 3,
                "gw": [1, 1, 1],
                "element": [7, 8, 9],
                "name": ["Youth"] * 3,
                "position": ["MID"] * 3,
                "team": ["Arsenal"] * 3,
                "bonus": [0] * 3,
                "minutes": [0, 0, 0],
                "round": [1] * 3,
                "total_points": [0] * 3,
                "value": [40] * 3,
            }
        ),
    )

    check_transfermarkt_freshness(db, SEASON)


def test_valuation_dates_are_read_as_dates(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """The age arithmetic works on a date, not a string.

    A stored ``value_date`` that came back as text would make the
    subtraction raise rather than compare, so this pins the type.
    """
    _seed_valuations(db, 0)
    newest = db.sql(
        f"SELECT MAX(value_date) FROM {TM_MARKET_VALUE.name}"
    ).fetchone()

    assert newest is not None
    assert isinstance(newest[0], (date, datetime))
