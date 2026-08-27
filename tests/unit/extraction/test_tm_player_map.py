"""Unit tests for the FPL-to-Transfermarkt player crosswalk."""

from datetime import date, datetime
from pathlib import Path

import duckdb
import polars as pl
import pytest

from fantasy_football.extraction import tm_player_map
from fantasy_football.extraction.tm_player_map import (
    build_player_map,
    load_club_map,
    load_overrides,
    store_player_map,
    unmatched_report,
)
from fantasy_football.storage.tables import (
    PLAYER_SEASON,
    PLAYER_WEEK,
    TM_PLAYER,
    TM_PLAYER_MAP,
    TM_TRANSFER,
)

CLUB_MAP = pl.DataFrame(
    {
        "tm_club": ["Manchester City", "Tottenham Hotspur"],
        "fpl_club": ["Man City", "Spurs"],
    }
)

NO_OVERRIDES = pl.DataFrame(
    {
        "player_code": pl.Series([], dtype=pl.Int64),
        "tm_player_id": pl.Series([], dtype=pl.Utf8),
        "fpl_name": pl.Series([], dtype=pl.Utf8),
        "tm_name": pl.Series([], dtype=pl.Utf8),
    }
)


def _add_fpl_player(
    connection: duckdb.DuckDBPyConnection,
    player_code: int,
    first_name: str,
    second_name: str,
    birth_date: date | None,
    team: str = "Man City",
    season: str = "2026-27",
) -> None:
    """Insert one FPL player, with a player-week row giving them a club."""
    element = player_code % 1000
    PLAYER_SEASON.append(
        connection,
        PLAYER_SEASON.conform(
            pl.DataFrame(
                {
                    "season": [season],
                    "element": [element],
                    "player_code": [player_code],
                    "first_name": [first_name],
                    "second_name": [second_name],
                    "birth_date": [birth_date],
                }
            )
        ),
    )
    PLAYER_WEEK.append(
        connection,
        PLAYER_WEEK.conform(
            pl.DataFrame(
                {
                    "season": [season],
                    "gw": [1],
                    "element": [element],
                    "name": [f"{first_name} {second_name}"],
                    "team": [team],
                }
            )
        ),
    )


def _add_tm_player(
    connection: duckdb.DuckDBPyConnection,
    tm_player_id: str,
    name: str,
    dob_date: date | None,
    team: str = "Manchester City",
) -> None:
    """Insert one Transfermarkt player."""
    TM_PLAYER.append(
        connection,
        TM_PLAYER.conform(
            pl.DataFrame(
                {
                    "tm_player_id": [tm_player_id],
                    "name": [name],
                    "dob_date": [dob_date],
                    "team": [team],
                    "scraped_at": [datetime(2026, 8, 27, 12, 0)],
                }
            )
        ),
    )


def _build(
    connection: duckdb.DuckDBPyConnection,
    overrides: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Run the waterfall with the test club map and no committed files."""
    return build_player_map(
        connection,
        overrides=NO_OVERRIDES if overrides is None else overrides,
        club_map=CLUB_MAP,
    )


def test_committed_club_map_maps_transfermarkt_to_fpl_names() -> None:
    """The shipped club file reads and carries both columns."""
    frame = load_club_map()
    assert frame.columns == ["tm_club", "fpl_club"]
    mapping = dict(zip(frame["tm_club"], frame["fpl_club"]))
    assert mapping["Manchester City"] == "Man City"


def test_committed_overrides_file_reads() -> None:
    """The shipped override file parses, empty or not."""
    assert load_overrides().columns == [
        "player_code",
        "tm_player_id",
        "fpl_name",
        "tm_name",
    ]


def _club_map_file(directory: Path) -> Path:
    """Write the test club map where a default-path caller will find it."""
    path = directory / "clubs.csv"
    CLUB_MAP.write_csv(path)
    return path


def _override_file(directory: Path) -> Path:
    """Write an empty override file for a default-path caller."""
    path = directory / "overrides.csv"
    NO_OVERRIDES.write_csv(path)
    return path


def _write_overrides(path: Path, rows: dict[str, list[object]]) -> Path:
    """Write an override CSV to a temporary path."""
    pl.DataFrame(rows).write_csv(path)
    return path


def test_load_overrides_rejects_a_repeated_player_code(
    tmp_path: Path,
) -> None:
    """Two Transfermarkt ids for one player is a one-to-many map."""
    path = _write_overrides(
        tmp_path / "o.csv",
        {
            "player_code": [1, 1],
            "tm_player_id": ["a", "b"],
            "fpl_name": ["x", "x"],
            "tm_name": ["x", "x"],
        },
    )
    with pytest.raises(ValueError, match="player_code"):
        load_overrides(path)


def test_load_overrides_rejects_a_repeated_tm_player_id(
    tmp_path: Path,
) -> None:
    """Two players sharing one Transfermarkt id would fan out the join."""
    path = _write_overrides(
        tmp_path / "o.csv",
        {
            "player_code": [1, 2],
            "tm_player_id": ["a", "a"],
            "fpl_name": ["x", "y"],
            "tm_name": ["x", "x"],
        },
    )
    with pytest.raises(ValueError, match="tm_player_id"):
        load_overrides(path)


def test_exact_name_and_dob_match(db: duckdb.DuckDBPyConnection) -> None:
    """The first rung claims an exact name on a shared birthday."""
    _add_fpl_player(db, 111, "Erling", "Haaland", date(2000, 7, 21))
    _add_tm_player(db, "418560", "Erling Haaland", date(2000, 7, 21))
    matches = _build(db)
    assert matches["match_rung"].to_list() == ["dob_exact_name"]
    assert matches["tm_player_id"].to_list() == ["418560"]


def test_accents_and_case_do_not_block_an_exact_match(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Normalisation strips the accent Transfermarkt keeps."""
    _add_fpl_player(db, 222, "Marc", "Guehi", date(2000, 7, 13))
    _add_tm_player(db, "413338", "Marc Guéhi", date(2000, 7, 13))
    assert _build(db)["match_rung"].to_list() == ["dob_exact_name"]


def test_a_similar_name_on_a_shared_birthday_matches(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A transliteration clears the fuzzy rung."""
    _add_fpl_player(db, 333, "Emiliano", "Martinez", date(1992, 9, 2))
    _add_tm_player(db, "111905", "Emiliano Martínez Romero", date(1992, 9, 2))
    matches = _build(db)
    assert matches["match_rung"].to_list() == ["dob_fuzzy_name"]
    assert matches["match_score"][0] >= 0.85


def test_a_different_person_on_a_shared_birthday_stays_unmatched(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A shared birthday alone is never enough."""
    _add_fpl_player(db, 444, "Bukayo", "Saka", date(2001, 9, 5))
    _add_tm_player(db, "999999", "Wojciech Szczesny", date(2001, 9, 5))
    assert _build(db).is_empty()


def test_a_shared_surname_matches_when_the_full_name_does_not(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """The surname rung catches a name the two sources shorten differently."""
    _add_fpl_player(db, 555, "Bobby", "De Cordova-Reid", date(1993, 2, 2))
    _add_tm_player(db, "200104", "Bobby Reid", date(1993, 2, 2))
    assert _build(db)["match_rung"].to_list() == ["dob_surname"]


def test_two_near_candidates_on_one_birthday_are_left_to_the_human(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Ambiguity goes to the override file rather than to the better score."""
    _add_fpl_player(db, 666, "Danny", "Ings", date(1992, 7, 23))
    _add_tm_player(db, "1", "Dany Ings", date(1992, 7, 23))
    _add_tm_player(db, "2", "Danni Ings", date(1992, 7, 23))
    assert _build(db).is_empty()


def test_an_exact_name_wins_a_crowded_birthday(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """One exact name among the near misses is claimed without asking."""
    _add_fpl_player(db, 666, "Danny", "Ings", date(1992, 7, 23))
    _add_tm_player(db, "1", "Danny Ings", date(1992, 7, 23))
    _add_tm_player(db, "2", "Dany Ings", date(1992, 7, 23))
    matches = _build(db)
    assert matches["tm_player_id"].to_list() == ["1"]
    assert matches["match_rung"].to_list() == ["exact_name"]


def test_a_null_birth_date_falls_back_to_name_and_club(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """With no birthday to gate on, the club does the guarding."""
    _add_fpl_player(db, 777, "Kevin", "De Bruyne", None, team="Man City")
    _add_tm_player(
        db, "88755", "Kevin De Bruyne", None, team="Manchester City"
    )
    assert _build(db)["match_rung"].to_list() == ["club_exact_name"]


def test_a_sole_exact_name_matches_without_a_shared_club(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Being the only one of that name is evidence enough on its own."""
    _add_fpl_player(db, 888, "James", "Smith", None, team="Man City")
    _add_tm_player(db, "3", "James Smith", None, team="Tottenham Hotspur")
    assert _build(db)["match_rung"].to_list() == ["exact_name"]


def test_two_players_of_the_same_name_are_left_to_the_human(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """The exact-name rungs claim a name only when it is unique."""
    _add_fpl_player(db, 889, "James", "Smith", None)
    _add_tm_player(db, "4", "James Smith", None)
    _add_tm_player(db, "5", "James Smith", date(1999, 1, 1))
    assert _build(db).is_empty()


def test_an_exact_name_over_clashing_birthdays_is_flagged(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Still matched, but under the rung that says why to distrust it."""
    _add_fpl_player(db, 890, "James", "Smith", date(1990, 1, 1))
    _add_tm_player(db, "6", "James Smith", date(1995, 5, 5))
    matches = _build(db)
    assert matches["match_rung"].to_list() == ["exact_name_dob_conflict"]


def test_a_club_matches_through_a_transfer_row(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A club the player has since left still corroborates identity."""
    _add_fpl_player(db, 999, "Kyle", "Walker", None, team="Man City")
    _add_tm_player(db, "95424", "Kyle Walker", None, team="Burnley")
    TM_TRANSFER.append(
        db,
        TM_TRANSFER.conform(
            pl.DataFrame(
                {
                    "tm_player_id": ["95424"],
                    "transfer_seq": [0],
                    "joined_club": ["Manchester City"],
                }
            )
        ),
    )
    assert _build(db)["match_rung"].to_list() == ["club_exact_name"]


def test_an_override_beats_the_automatic_match(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """The hand-written file is the golden source, checked first."""
    _add_fpl_player(db, 1234, "Erling", "Haaland", date(2000, 7, 21))
    _add_tm_player(db, "418560", "Erling Haaland", date(2000, 7, 21))
    _add_tm_player(db, "correct", "Erling Braut Haaland", date(2000, 7, 21))
    overrides = pl.DataFrame(
        {
            "player_code": [1234],
            "tm_player_id": ["correct"],
            "fpl_name": ["Erling Haaland"],
            "tm_name": ["Erling Braut Haaland"],
        }
    )
    matches = _build(db, overrides=overrides)
    assert matches["match_rung"].to_list() == ["override"]
    assert matches["tm_player_id"].to_list() == ["correct"]


def test_an_overridden_transfermarkt_id_is_not_reused(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A claimed id is withdrawn from every later rung, keeping the map 1:1."""
    _add_fpl_player(db, 1, "Erling", "Haaland", date(2000, 7, 21))
    _add_fpl_player(db, 2, "Erling", "Haaland", date(2000, 7, 21))
    _add_tm_player(db, "418560", "Erling Haaland", date(2000, 7, 21))
    overrides = pl.DataFrame(
        {
            "player_code": [1],
            "tm_player_id": ["418560"],
            "fpl_name": ["Erling Haaland"],
            "tm_name": ["Erling Haaland"],
        }
    )
    matches = _build(db, overrides=overrides)
    assert matches["player_code"].to_list() == [1]
    assert not matches["tm_player_id"].is_duplicated().any()


def test_the_report_offers_candidates_in_the_override_schema(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """An unmatched player arrives with its best candidates, best first."""
    _add_fpl_player(db, 4321, "Bukayo", "Saka", date(2001, 9, 5))
    _add_tm_player(db, "far", "Wojciech Szczesny", date(1990, 4, 18))
    _add_tm_player(db, "near", "Bukayo Sako", date(1990, 4, 18))
    report = unmatched_report(db, _build(db))
    assert report.columns[:4] == [
        "player_code",
        "tm_player_id",
        "fpl_name",
        "tm_name",
    ]
    assert report["tm_player_id"].to_list() == ["near", "far"]


def test_a_matched_player_is_absent_from_the_report(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """The report is the to-do list, so a solved player leaves it."""
    _add_fpl_player(db, 111, "Erling", "Haaland", date(2000, 7, 21))
    _add_tm_player(db, "418560", "Erling Haaland", date(2000, 7, 21))
    assert unmatched_report(db, _build(db)).is_empty()


def test_storing_the_map_replaces_rather_than_accumulates(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """A rebuild reproduces the table, so a stale row cannot survive one."""
    _add_fpl_player(db, 111, "Erling", "Haaland", date(2000, 7, 21))
    _add_tm_player(db, "418560", "Erling Haaland", date(2000, 7, 21))
    matches = _build(db)
    store_player_map(db, matches)
    store_player_map(db, matches)
    assert TM_PLAYER_MAP.load(db).height == 1


def test_an_override_unblocks_the_other_player_on_that_birthday(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Resolving one of a pair leaves the other unambiguous."""
    _add_fpl_player(db, 10, "Danny", "Ings", date(1992, 7, 23))
    _add_fpl_player(db, 20, "Dany", "Ings", date(1992, 7, 23))
    _add_tm_player(db, "1", "Danny Ings", date(1992, 7, 23))
    _add_tm_player(db, "2", "Dany Ings", date(1992, 7, 23))
    overrides = pl.DataFrame(
        {
            "player_code": [10],
            "tm_player_id": ["1"],
            "fpl_name": ["Danny Ings"],
            "tm_name": ["Danny Ings"],
        }
    )
    matches = _build(db, overrides=overrides)
    assert matches["player_code"].to_list() == [10, 20]
    assert matches["tm_player_id"].to_list() == ["1", "2"]


def test_a_player_with_no_candidates_still_reaches_the_report(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """An empty Transfermarkt side must not hide the work still to do."""
    _add_fpl_player(db, 31, "Someone", "Unscraped", date(2004, 1, 1))
    report = unmatched_report(db, _build(db))
    assert report["player_code"].to_list() == [31]
    assert report["tm_player_id"].to_list() == [None]


def test_the_stored_map_rejects_a_repeated_transfermarkt_id(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """The one-to-one guarantee is the table's, not just the loader's."""
    rows = pl.DataFrame(
        {
            "player_code": [1, 2],
            "tm_player_id": ["same", "same"],
            "match_rung": ["override", "override"],
            "match_score": [None, None],
            "fpl_name": ["a", "b"],
            "tm_name": ["a", "b"],
        }
    )
    with pytest.raises(duckdb.ConstraintException):
        TM_PLAYER_MAP.append(db, TM_PLAYER_MAP.conform(rows))


def test_refresh_writes_the_map_and_the_candidate_report(
    db: duckdb.DuckDBPyConnection,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One call rebuilds the table and leaves the to-do list on disk."""
    monkeypatch.setattr(tm_player_map, "CLUBS_PATH", _club_map_file(tmp_path))
    monkeypatch.setattr(
        tm_player_map, "OVERRIDES_PATH", _override_file(tmp_path)
    )
    _add_fpl_player(db, 111, "Erling", "Haaland", date(2000, 7, 21))
    _add_fpl_player(db, 222, "Someone", "Unscraped", date(2004, 1, 1))
    _add_tm_player(db, "418560", "Erling Haaland", date(2000, 7, 21))
    report_path = tmp_path / "candidates.csv"

    matches = tm_player_map.refresh_player_map(db, report_path=report_path)

    assert matches["player_code"].to_list() == [111]
    assert TM_PLAYER_MAP.load(db).height == 1
    assert pl.read_csv(report_path)["player_code"].to_list() == [222]


def test_the_report_honours_the_candidate_limit(
    db: duckdb.DuckDBPyConnection,
) -> None:
    """Only the best few candidates are offered per unmatched player."""
    _add_fpl_player(db, 4321, "Bukayo", "Saka", date(2001, 9, 5))
    for index in range(3):
        _add_tm_player(
            db, f"tm{index}", f"Bukayo Sak{index}", date(1990, 4, 18)
        )
    report = unmatched_report(db, _build(db), candidates=1)
    assert report.height == 1
