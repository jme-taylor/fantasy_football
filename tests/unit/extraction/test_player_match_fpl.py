"""Tests for the Vaastav per-fixture player stats loader."""

import polars as pl

from fantasy_football.extraction.player_match_fpl import (
    VaastavMatchLoader,
    shape_merged_gw,
)
from fantasy_football.storage.tables import PLAYER_MATCH_FPL


def _source_frame() -> pl.DataFrame:
    """Return a two-row frame shaped like a Vaastav merged_gw.csv."""
    return pl.DataFrame(
        {
            "GW": [1, 1],
            "element": [10, 11],
            "fixture": [3, 3],
            "name": ["A B", "C D"],
            "minutes": [90, 45],
            "goals_scored": [1, 0],
            "total_points": [8, 2],
            "opponent_team": [5, 5],
            "was_home": [True, True],
            "kickoff_time": ["2016-08-13T11:30:00Z", "2016-08-13T11:30:00Z"],
        }
    )


def test_shape_merged_gw_renames_gw_and_adds_season():
    """The source's GW column becomes gw and the season is stamped on."""
    shaped = shape_merged_gw(_source_frame(), "2016-17")
    assert "gw" in shaped.columns
    assert "GW" not in shaped.columns
    assert shaped["season"].unique().to_list() == ["2016-17"]


def test_shape_merged_gw_drops_verbatim_duplicates():
    """A byte-for-byte repeated row is dropped, not summed."""
    doubled = pl.concat([_source_frame(), _source_frame().head(1)])
    assert shape_merged_gw(doubled, "2025-26").height == 2


def test_shape_merged_gw_keeps_both_legs_of_a_double_gameweek():
    """Two fixtures in one gameweek stay as two rows."""
    frame = _source_frame().head(1)
    second = frame.with_columns(
        pl.lit(4, dtype=pl.Int64).alias("fixture"),
        pl.lit(9, dtype=pl.Int64).alias("opponent_team"),
    )
    shaped = shape_merged_gw(pl.concat([frame, second]), "2025-26")
    assert shaped.height == 2
    assert sorted(shaped["fixture"].to_list()) == [3, 4]


def test_load_season_conforms_missing_columns_to_the_schema(mocker):
    """A 2016-17 frame lacking expected_goals still coerces to the schema."""
    loader = VaastavMatchLoader()
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    frame = loader.load_season("2016-17")
    assert frame.columns == PLAYER_MATCH_FPL.columns
    assert frame["expected_goals"].to_list() == [None, None]
    assert frame["goals_scored"].to_list() == [1, 0]


def test_load_season_reads_legacy_seasons_with_lossy_encoding(mocker):
    """The non-UTF-8 2016-19 files are read leniently."""
    loader = VaastavMatchLoader()
    read = mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    loader.load_season("2016-17")
    assert read.call_args.kwargs["encoding"] == "utf8-lossy"


def test_load_season_warns_about_undeclared_source_columns(mocker, caplog):
    """A new upstream column is logged rather than silently dropped."""
    loader = VaastavMatchLoader()
    mocker.patch.object(
        loader.extractor,
        "_read_csv",
        return_value=_source_frame().with_columns(
            pl.lit(1).alias("brand_new")
        ),
    )
    with caplog.at_level("WARNING"):
        loader.load_season("2016-17")
    assert "brand_new" in caplog.text


def test_available_seasons_reads_the_repo_tree(mocker):
    """Seasons are discovered, not hardcoded, so a new one loads itself."""
    loader = VaastavMatchLoader()
    mocker.patch.object(
        loader.extractor.api_client,
        "get_all_repo_files",
        return_value={
            "tree": [
                {"path": "data/2016-17/gws/merged_gw.csv"},
                {"path": "data/2025-26/gws/merged_gw.csv"},
                {"path": "data/2025-26/players_raw.csv"},
            ]
        },
    )
    assert loader.available_seasons() == ["2016-17", "2025-26"]


def test_load_writes_completed_seasons_and_skips_them_on_rerun(mocker, db):
    """Completed seasons are immutable: a second run is a no-op."""
    loader = VaastavMatchLoader()
    mocker.patch.object(loader, "available_seasons", return_value=["2016-17"])
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    loader.load(db, current_season="2026-27")
    loader.load(db, current_season="2026-27")
    stored = PLAYER_MATCH_FPL.load(db)
    assert stored.height == 2


def test_load_tolerates_the_current_season_being_absent(mocker, db):
    """Vaastav has no 2026-27; that is normal, not an error."""
    loader = VaastavMatchLoader()
    mocker.patch.object(loader, "available_seasons", return_value=["2016-17"])
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    loader.load(db, current_season="2026-27")
    assert PLAYER_MATCH_FPL.seasons_present(db) == {"2016-17"}


def test_load_upserts_the_current_season_when_present(mocker, db):
    """When the source does carry the current season, it is refreshed."""
    loader = VaastavMatchLoader()
    mocker.patch.object(loader, "available_seasons", return_value=["2025-26"])
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    loader.load(db, current_season="2025-26")
    mocker.patch.object(
        loader.extractor,
        "_read_csv",
        return_value=_source_frame().with_columns(
            pl.lit(99).alias("total_points")
        ),
    )
    loader.load(db, current_season="2025-26")
    stored = PLAYER_MATCH_FPL.load(db)
    assert stored.height == 2
    assert stored["total_points"].unique().to_list() == [99]


def test_load_leaves_a_populated_current_season_untouched_on_empty_fetch(
    mocker, db, caplog
):
    """An empty current-season fetch must not wipe already-stored rows.

    ``upsert_current`` deletes a season's rows before inserting, so a
    naive upsert of an empty frame would silently empty the table.
    """
    loader = VaastavMatchLoader()
    mocker.patch.object(loader, "available_seasons", return_value=["2025-26"])
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    loader.load(db, current_season="2025-26")
    assert PLAYER_MATCH_FPL.load(db).height == 2

    mocker.patch.object(
        loader.extractor,
        "_read_csv",
        return_value=_source_frame().filter(pl.lit(False)),
    )
    with caplog.at_level("WARNING"):
        loader.load(db, current_season="2025-26")

    assert PLAYER_MATCH_FPL.load(db).height == 2
    assert "2025-26" in caplog.text
