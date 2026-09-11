"""Unit tests for Transfermarkt parsing, shaping and storage."""

from datetime import date, datetime

import duckdb
import pandas as pd
import polars as pl
import pytest

from fantasy_football.extraction.transfermarkt import (
    classify_fee,
    collect_player_links,
    parse_money,
    parse_tm_date,
    pending_player_links,
    player_id_from_link,
    season_to_tm_year,
    shape_player,
    shape_player_season,
    store_player,
)
from fantasy_football.storage.tables import (
    TM_MARKET_VALUE,
    TM_PLAYER,
    TM_PLAYER_SEASON,
    TM_TRANSFER,
)


def test_parse_money_reads_millions() -> None:
    """A millions suffix scales the amount to whole euros."""
    assert parse_money("€35.00m") == 35_000_000


def test_parse_money_reads_thousands_and_billions() -> None:
    """Thousands and billions suffixes scale the amount too."""
    assert parse_money("€900k") == 900_000
    assert parse_money("€1.20bn") == 1_200_000_000


def test_parse_money_reads_bare_units() -> None:
    """An amount with no suffix is already in whole euros."""
    assert parse_money("€500") == 500


def test_parse_money_finds_the_amount_inside_a_labelled_fee() -> None:
    """A labelled loan fee still yields its amount."""
    assert parse_money("Loan fee: €2.00m") == 2_000_000


def test_parse_money_ignores_the_currency_symbol() -> None:
    """Non-euro symbols are ignored; only the number is read."""
    assert parse_money("£4.50m") == 4_500_000


def test_parse_money_returns_none_when_there_is_no_amount() -> None:
    """Fee strings carrying no number parse to None."""
    for value in (
        "free transfer",
        "loan transfer",
        "End of loan",
        "-",
        "?",
        "",
    ):
        assert parse_money(value) is None
    assert parse_money(None) is None


def test_parse_money_returns_none_for_punctuation_that_is_not_a_number() -> (
    None
):
    """Punctuation alone is not an amount."""
    assert parse_money("€.") is None


def test_classify_fee_reads_a_free_transfer() -> None:
    """Punctuation alone is not an amount."""
    assert classify_fee("free transfer") == "free"


def test_classify_fee_reads_a_loan() -> None:
    """Loans are classified as loan, with or without a fee."""
    assert classify_fee("loan transfer") == "loan"
    assert classify_fee("Loan fee: €2.00m") == "loan"


def test_classify_fee_reads_the_end_of_a_loan() -> None:
    """End of loan is distinguished from a loan starting."""
    assert classify_fee("End of loan") == "loan_end"


def test_classify_fee_reads_a_paid_transfer() -> None:
    """A bare amount is a paid transfer."""
    assert classify_fee("€35.00m") == "transfer"


def test_classify_fee_falls_back_to_unknown() -> None:
    """Placeholders and blanks classify as unknown."""
    for value in ("-", "?", "", None):
        assert classify_fee(value) == "unknown"


def test_parse_tm_date_reads_the_player_page_format() -> None:
    """The .us player page date format parses."""
    assert parse_tm_date("Jul 1, 2016") == date(2016, 7, 1)


def test_parse_tm_date_reads_the_market_value_endpoint_format() -> None:
    """The market value endpoint date format parses."""
    assert parse_tm_date("Dec 15, 2023") == date(2023, 12, 15)


def test_parse_tm_date_tolerates_surrounding_whitespace() -> None:
    """Surrounding whitespace does not defeat the parse."""
    assert parse_tm_date("  Jan 3, 2020  ") == date(2020, 1, 3)


def test_parse_tm_date_reads_the_dotted_european_format() -> None:
    """The dotted European date format parses."""
    assert parse_tm_date("15.12.2023") == date(2023, 12, 15)


def test_parse_tm_date_returns_none_when_there_is_no_date() -> None:
    """Placeholders and partial dates parse to None."""
    for value in ("-", "?", "", "Jul 2016", None):
        assert parse_tm_date(value) is None


SCRAPED_AT = datetime(2026, 8, 27, 12, 0, 0)


def _scraped_player(
    market_value_history: pd.DataFrame | None = None,
    transfer_history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a scrape_player-shaped 1-row frame for a single player."""
    if market_value_history is None:
        market_value_history = pd.DataFrame(
            {"date": ["Dec 15, 2023"], "value": [35000000]}
        )
    if transfer_history is None:
        transfer_history = pd.DataFrame(
            [
                [
                    "23/24",
                    "Jul 1, 2023",
                    "Burnley",
                    "Newcastle",
                    "€8.00m",
                    "€10.00m",
                ]
            ],
            columns=pd.Index(
                ["Season", "Date", "Left", "Joined", "MV", "Fee"]
            ),
        )
    player = pd.Series(dtype=object)
    player["Name"] = "Nick Pope"
    player["ID"] = "112988"
    player["Value"] = "€12.00m"
    player["Value last updated"] = "Dec 15, 2023"
    player["DOB"] = "Apr 19, 1992"
    player["Age"] = 34
    player["Height (m)"] = 1.91
    player["Nationality"] = "England"
    player["Citizenship"] = ["England"]
    player["Position"] = "Goalkeeper"
    player["Other positions"] = None
    player["Team"] = "Newcastle United"
    player["Last club"] = None
    player["Since"] = None
    player["Joined"] = "Jul 1, 2022"
    player["Contract expiration"] = "Jun 30, 2026"
    player["Market value history"] = market_value_history
    player["Transfer history"] = transfer_history
    return player.to_frame().T


def test_shape_player_keeps_the_raw_string_and_the_parsed_value() -> None:
    """Raw strings survive alongside their parsed siblings."""
    player, _, _ = shape_player(_scraped_player(), SCRAPED_AT)
    row = player.to_dicts()[0]
    assert row["tm_player_id"] == "112988"
    assert row["value"] == "€12.00m"
    assert row["value_eur"] == 12_000_000
    assert row["joined"] == "Jul 1, 2022"
    assert row["joined_date"] == date(2022, 7, 1)
    assert row["scraped_at"] == SCRAPED_AT


def test_shape_player_joins_citizenships_with_a_pipe() -> None:
    """A citizenship list becomes one pipe-delimited string."""
    scraped = _scraped_player()
    scraped.at[0, "Citizenship"] = ["England", "Ireland"]
    player, _, _ = shape_player(scraped, SCRAPED_AT)
    assert player.to_dicts()[0]["citizenship"] == "England|Ireland"


def test_shape_player_numbers_transfers_newest_first() -> None:
    """transfer_seq follows scrape order, newest at zero."""
    transfers = pd.DataFrame(
        [
            [
                "23/24",
                "Jul 1, 2023",
                "Burnley",
                "Newcastle",
                "€8.00m",
                "€10.00m",
            ],
            [
                "16/17",
                "Jul 1, 2016",
                "Charlton",
                "Burnley",
                "€1.50m",
                "free transfer",
            ],
        ],
        columns=pd.Index(["Season", "Date", "Left", "Joined", "MV", "Fee"]),
    )
    _, transfer, _ = shape_player(
        _scraped_player(transfer_history=transfers), SCRAPED_AT
    )
    rows = transfer.sort("transfer_seq").to_dicts()
    assert [row["transfer_seq"] for row in rows] == [0, 1]
    assert rows[0]["joined_club"] == "Newcastle"
    assert rows[0]["fee_eur"] == 10_000_000
    assert rows[0]["fee_type"] == "transfer"
    assert rows[1]["fee_eur"] is None
    assert rows[1]["fee_type"] == "free"
    assert rows[1]["transfer_date_parsed"] == date(2016, 7, 1)


def test_shape_player_keeps_market_values_already_typed() -> None:
    """Market values arrive typed and keep their raw date."""
    _, _, market_value = shape_player(_scraped_player(), SCRAPED_AT)
    row = market_value.to_dicts()[0]
    assert row["value_date"] == date(2023, 12, 15)
    assert row["value_date_raw"] == "Dec 15, 2023"
    assert row["value_eur"] == 35_000_000


def test_shape_player_drops_market_values_whose_date_will_not_parse() -> None:
    """A market value with no parseable date cannot be keyed, so it is dropped."""
    history = pd.DataFrame(
        {"date": ["Dec 15, 2023", "-"], "value": [35000000, 40000000]}
    )
    _, _, market_value = shape_player(
        _scraped_player(market_value_history=history), SCRAPED_AT
    )
    assert market_value.height == 1
    assert market_value.to_dicts()[0]["value_date"] == date(2023, 12, 15)


def test_shape_player_handles_a_player_with_no_history_at_all() -> None:
    """Absent histories yield empty frames in the stored schema."""
    scraped = _scraped_player(
        market_value_history=None,
        transfer_history=pd.DataFrame(
            columns=pd.Index(
                ["Season", "Date", "Left", "Joined", "MV", "Fee"]
            ),
        ),
    )
    scraped.at[0, "Market value history"] = None
    player, transfer, market_value = shape_player(scraped, SCRAPED_AT)
    assert player.height == 1
    assert transfer.height == 0
    assert market_value.height == 0
    assert transfer.schema == dict(TM_TRANSFER.schema)
    assert market_value.schema == dict(TM_MARKET_VALUE.schema)


def test_shape_player_records_the_link_it_was_scraped_from() -> None:
    """The scrape does not return its own URL, so the caller supplies it."""
    player, _, _ = shape_player(
        _scraped_player(),
        SCRAPED_AT,
        player_link="https://www.transfermarkt.us/nick-pope/profil/spieler/112988",
    )
    assert player.to_dicts()[0]["player_link"] == (
        "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    )


def test_season_to_tm_year_converts_to_transfermarkt_form() -> None:
    """Short-form seasons convert to Transfermarkt's YY/YY."""
    assert season_to_tm_year("2016-17") == "16/17"
    assert season_to_tm_year("2026-27") == "26/27"


def test_player_id_from_link_takes_the_trailing_url_segment() -> None:
    """The player id is the last segment of the URL."""
    link = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    assert player_id_from_link(link) == "112988"


def test_player_id_from_link_ignores_a_trailing_slash() -> None:
    """A trailing slash does not become part of the id."""
    link = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988/"
    assert player_id_from_link(link) == "112988"


def test_shape_player_season_keys_links_by_season_and_id() -> None:
    """Season links become rows keyed by season and player id."""
    links = [
        "https://www.transfermarkt.us/nick-pope/profil/spieler/112988",
        "https://www.transfermarkt.us/kieran-trippier/profil/spieler/94612",
    ]
    frame = shape_player_season("2025-26", links)
    rows = frame.sort("tm_player_id").to_dicts()
    assert [row["tm_player_id"] for row in rows] == ["112988", "94612"]
    assert {row["season"] for row in rows} == {"2025-26"}
    assert rows[0]["player_link"] == links[0]


def test_shape_player_season_drops_repeated_links() -> None:
    """A link repeated within a season yields one row."""
    link = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    assert shape_player_season("2025-26", [link, link]).height == 1


def test_shape_player_season_handles_an_empty_season() -> None:
    """An empty link list yields an empty frame in schema."""
    frame = shape_player_season("2025-26", [])
    assert frame.height == 0
    assert frame.schema == dict(TM_PLAYER_SEASON.schema)


def _store_one(db: duckdb.DuckDBPyConnection, link: str) -> None:
    """Scrape-and-store one player from a canned scrape."""
    player, transfers, market_values = shape_player(
        _scraped_player(), SCRAPED_AT, player_link=link
    )
    store_player(db, player, transfers, market_values)


def test_store_player_writes_all_three_frames(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """One player lands in all three tables."""
    _store_one(
        db, "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    )
    assert TM_PLAYER.load(db).height == 1
    assert TM_TRANSFER.load(db).height == 1
    assert TM_MARKET_VALUE.load(db).height == 1


def test_store_player_is_idempotent(db: "duckdb.DuckDBPyConnection") -> None:
    """Re-storing a player replaces rather than duplicates."""
    link = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    _store_one(db, link)
    _store_one(db, link)
    assert TM_PLAYER.load(db).height == 1
    assert TM_TRANSFER.load(db).height == 1
    assert TM_MARKET_VALUE.load(db).height == 1


def test_pending_player_links_skips_players_already_stored(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """The anti-join is what makes the scrape resumable."""
    pope = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    trippier = (
        "https://www.transfermarkt.us/kieran-trippier/profil/spieler/94612"
    )
    TM_PLAYER_SEASON.append(
        db, shape_player_season("2025-26", [pope, trippier])
    )
    _store_one(db, pope)
    assert pending_player_links(db) == [trippier]


def test_pending_player_links_returns_everything_when_refreshing(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A refresh ignores what is already stored."""
    pope = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    trippier = (
        "https://www.transfermarkt.us/kieran-trippier/profil/spieler/94612"
    )
    TM_PLAYER_SEASON.append(
        db, shape_player_season("2025-26", [pope, trippier])
    )
    _store_one(db, pope)
    assert sorted(pending_player_links(db, refresh=True)) == sorted(
        [pope, trippier]
    )


def test_pending_player_links_deduplicates_a_player_across_seasons(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A player in several seasons is scraped once."""
    pope = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    for season in ("2024-25", "2025-26"):
        TM_PLAYER_SEASON.append(db, shape_player_season(season, [pope]))
    assert pending_player_links(db) == [pope]


def test_pending_player_links_honours_a_limit(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A limit caps how many players a run scrapes."""
    links = [
        f"https://www.transfermarkt.us/player-{index}/profil/spieler/{index}"
        for index in range(5)
    ]
    TM_PLAYER_SEASON.append(db, shape_player_season("2025-26", links))
    assert len(pending_player_links(db, limit=2)) == 2


def test_shape_player_stores_a_missing_height_as_null_not_nan() -> None:
    """A NaN height must land as NULL, since NaN is not a missing value in SQL."""
    scraped = _scraped_player()
    scraped.at[0, "Height (m)"] = float("nan")
    player, _, _ = shape_player(scraped, SCRAPED_AT)
    assert player.to_dicts()[0]["height_m"] is None


def test_store_player_leaves_no_trace_when_a_history_write_fails(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A half-written player must stay pending, not look scraped."""
    player, transfers, _ = shape_player(
        _scraped_player(), SCRAPED_AT, player_link="link"
    )
    # Two market values on one date breach the primary key, standing in for
    # any mid-write failure.
    broken = pl.DataFrame(
        {
            "tm_player_id": ["112988", "112988"],
            "value_date": [date(2023, 12, 15), date(2023, 12, 15)],
            "value_date_raw": ["Dec 15, 2023", "Dec 15, 2023"],
            "value_eur": [35000000, 40000000],
        },
        schema=TM_MARKET_VALUE.schema,
    )
    with pytest.raises(duckdb.Error):
        store_player(db, player, transfers, broken)
    assert TM_PLAYER.load(db).is_empty()


def test_shape_player_keeps_one_market_value_per_date() -> None:
    """value_date is half the primary key, so repeats cannot both be stored."""
    history = pd.DataFrame(
        {
            "date": ["Dec 15, 2023", "Dec 15, 2023"],
            "value": [35000000, 40000000],
        }
    )
    _, _, market_value = shape_player(
        _scraped_player(market_value_history=history), SCRAPED_AT
    )
    assert market_value.height == 1


def test_shape_player_sorts_citizenships_so_a_rescrape_matches() -> None:
    """ScraperFC builds citizenship from a set, whose order is not stable."""
    scraped = _scraped_player()
    scraped.at[0, "Citizenship"] = ["Ireland", "England"]
    player, _, _ = shape_player(scraped, SCRAPED_AT)
    assert player.to_dicts()[0]["citizenship"] == "England|Ireland"


def test_shape_player_falls_back_to_the_link_for_the_id() -> None:
    """A trailing slash empties ScraperFC's id, which would strand the player."""
    scraped = _scraped_player()
    scraped.at[0, "ID"] = ""
    player, _, _ = shape_player(
        scraped,
        SCRAPED_AT,
        player_link="https://www.transfermarkt.us/nick-pope/profil/spieler/112988/",
    )
    assert player.to_dicts()[0]["tm_player_id"] == "112988"


def test_parse_money_reads_dot_grouped_thousands() -> None:
    """Two dots can only be grouping; one is the decimal separator."""
    assert parse_money("€1.500.000") == 1_500_000
    assert parse_money("€1.20m") == 1_200_000


class _StubTransfermarkt:
    """A ScraperFC stand-in returning canned links per Transfermarkt year."""

    def __init__(self, links_by_year: dict[str, list[str]]) -> None:
        self.links_by_year = links_by_year

    def get_player_links(self, year: str, league: str) -> list[str]:
        """Return the canned links for a year, raising if none are canned."""
        if year not in self.links_by_year:
            raise RuntimeError(f"no links for {year}")
        return self.links_by_year[year]


def test_collect_player_links_stores_a_season(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Each season's squad links land as tm_player_season rows."""
    link = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    collect_player_links(
        db, ["2025-26"], _StubTransfermarkt({"25/26": [link]})
    )
    assert TM_PLAYER_SEASON.load(db).height == 1


def test_collect_player_links_keeps_stored_links_when_a_scrape_returns_nothing(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A Cloudflare block yields an empty list, which must not wipe the season."""
    link = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    collect_player_links(
        db, ["2025-26"], _StubTransfermarkt({"25/26": [link]})
    )
    collect_player_links(db, ["2025-26"], _StubTransfermarkt({"25/26": []}))
    assert TM_PLAYER_SEASON.load(db).height == 1


def test_collect_player_links_carries_on_past_a_failed_season(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """One unpublished season must not abort collection for the rest."""
    link = "https://www.transfermarkt.us/nick-pope/profil/spieler/112988"
    collect_player_links(
        db, ["2025-26", "2026-27"], _StubTransfermarkt({"25/26": [link]})
    )
    assert TM_PLAYER_SEASON.load(db).height == 1
