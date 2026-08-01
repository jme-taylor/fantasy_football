"""Tests for the FCI per-fixture Opta stats loader."""

import polars as pl
import pytest

from fantasy_football.extraction.player_match_opta import (
    FciMatchStatsLoader,
    add_competition,
    parse_competition,
)
from fantasy_football.storage.tables import PLAYER_MATCH_OPTA


def _source_frame() -> pl.DataFrame:
    """Return a two-row frame shaped like an FCI playermatchstats.csv."""
    return pl.DataFrame(
        {
            "player_id": [443, 16],
            "match_id": [
                "25-26-prem-manchester-united-v-arsenal",
                "25-26-champions-arsenal-v-inter",
            ],
            "minutes_played": [79, 90],
            "goals": [0, 1],
            "assists": [1, 0],
            "xg": [0.09, 0.4],
        }
    )


@pytest.mark.parametrize(
    "match_id,expected",
    [
        ("25-26-prem-manchester-united-v-arsenal", "prem"),
        ("25-26-efl-arsenal-v-brighton", "efl"),
        ("25-26-champions-arsenal-v-inter", "champions"),
        ("25-26-europa-villa-v-roma", "europa"),
        ("25-26-conference-chelsea-v-gent", "conference"),
    ],
)
def test_parse_competition_reads_the_slug_prefix(match_id, expected):
    """Each observed competition token is recognised."""
    assert parse_competition(match_id) == expected


def test_parse_competition_returns_none_for_an_unparseable_slug():
    """A slug that does not match the pattern yields None, not a crash."""
    assert parse_competition("not-a-match-id") is None


def test_parse_competition_keeps_an_unrecognised_token():
    """A new competition is stored as-is rather than dropped."""
    assert parse_competition("26-27-supercup-arsenal-v-psg") == "supercup"


def test_add_competition_populates_the_column():
    """Every row gains the competition parsed from its match_id."""
    result = add_competition(_source_frame())
    assert result["competition"].to_list() == ["prem", "champions"]


def test_load_season_keeps_every_competition(mocker):
    """European and cup rows are stored, not filtered away."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(
        loader, "matchstats_paths", return_value={1: "data/x/GW1/f.csv"}
    )
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    frame = loader.load_season("2025-26")
    assert sorted(frame["competition"].to_list()) == ["champions", "prem"]


def test_load_season_renames_player_id_to_element(mocker):
    """FCI's player_id is the FPL element and is stored under that name."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(
        loader, "matchstats_paths", return_value={1: "data/x/GW1/f.csv"}
    )
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    frame = loader.load_season("2025-26")
    assert frame.columns == PLAYER_MATCH_OPTA.columns
    assert sorted(frame["element"].to_list()) == [16, 443]


def test_load_season_stamps_gameweek_from_the_path(mocker):
    """The gameweek comes from the folder, not from the file contents."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(
        loader,
        "matchstats_paths",
        return_value={7: "data/x/GW7/f.csv"},
    )
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    frame = loader.load_season("2025-26")
    assert frame["gw"].unique().to_list() == [7]


def test_load_season_conforms_columns_absent_in_2024_25(mocker):
    """2024-25 lacks nine columns; they arrive as typed nulls."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(
        loader, "matchstats_paths", return_value={1: "data/x/GW1/f.csv"}
    )
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    frame = loader.load_season("2024-25")
    assert frame["defensive_contributions"].to_list() == [None, None]
    assert frame["corners"].to_list() == [None, None]


def test_load_season_warns_about_undeclared_source_columns(mocker, caplog):
    """A new upstream column such as corners was is logged when unknown."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(
        loader, "matchstats_paths", return_value={1: "data/x/GW1/f.csv"}
    )
    mocker.patch.object(
        loader.extractor,
        "_read_csv",
        return_value=_source_frame().with_columns(
            pl.lit(1).alias("brand_new")
        ),
    )
    with caplog.at_level("WARNING"):
        loader.load_season("2025-26")
    assert "brand_new" in caplog.text


def test_matchstats_paths_handles_the_2024_25_layout(mocker):
    """2024-25 stores files under playermatchstats/, not By Gameweek/."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(
        loader.extractor.api_client,
        "get_all_repo_files",
        return_value={
            "tree": [
                {
                    "path": "data/2024-2025/playermatchstats/GW1/"
                    "playermatchstats.csv"
                },
                {
                    "path": "data/2024-2025/playermatchstats/GW10/"
                    "playermatchstats.csv"
                },
            ]
        },
    )
    paths = loader.matchstats_paths("2024-25")
    assert sorted(paths) == [1, 10]


def test_matchstats_paths_handles_the_by_gameweek_layout(mocker):
    """2025-26 onwards store files under By Gameweek/."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(
        loader.extractor.api_client,
        "get_all_repo_files",
        return_value={
            "tree": [
                {
                    "path": "data/2025-2026/By Gameweek/GW3/"
                    "playermatchstats.csv"
                }
            ]
        },
    )
    assert loader.matchstats_paths("2025-26") == {
        3: "data/2025-2026/By Gameweek/GW3/playermatchstats.csv"
    }


def test_load_skips_completed_seasons_on_rerun(mocker, db):
    """Completed FCI seasons are immutable."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(loader, "available_seasons", return_value=["2024-25"])
    mocker.patch.object(
        loader, "matchstats_paths", return_value={1: "data/x/GW1/f.csv"}
    )
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    loader.load(db, current_season="2026-27")
    loader.load(db, current_season="2026-27")
    assert PLAYER_MATCH_OPTA.load(db).height == 2


def test_load_upserts_the_current_season(mocker, db):
    """The current season is replaced wholesale on each run."""
    loader = FciMatchStatsLoader()
    mocker.patch.object(loader, "available_seasons", return_value=["2026-27"])
    mocker.patch.object(
        loader, "matchstats_paths", return_value={1: "data/x/GW1/f.csv"}
    )
    mocker.patch.object(
        loader.extractor, "_read_csv", return_value=_source_frame()
    )
    loader.load(db, current_season="2026-27")
    mocker.patch.object(
        loader.extractor,
        "_read_csv",
        return_value=_source_frame().with_columns(pl.lit(5).alias("goals")),
    )
    loader.load(db, current_season="2026-27")
    stored = PLAYER_MATCH_OPTA.load(db)
    assert stored.height == 2
    assert stored["goals"].unique().to_list() == [5]
