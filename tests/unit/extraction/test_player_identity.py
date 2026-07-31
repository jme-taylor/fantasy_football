from datetime import date

import polars as pl
import pytest
import requests

from fantasy_football.extraction.player_identity import (
    build_player_season_from_fci,
    build_player_season_from_vaastav,
)
from fantasy_football.storage.tables import PLAYER_SEASON


def _vaastav_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2],
            "code": [111, 222],
            "web_name": ["Salah", "Saka"],
            "first_name": ["Mohamed", "Bukayo"],
            "second_name": ["Salah", "Saka"],
            "element_type": [3, 3],
            "team_code": [14, 3],
        }
    )


def _fci_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_id": [1, 2],
            "player_code": [111, 222],
            "web_name": ["Salah", "Saka"],
            "first_name": ["Mohamed", "Bukayo"],
            "second_name": ["Salah", "Saka"],
            "position": ["Midfielder", "Goalkeeper"],
            "team_code": [14, 3],
        }
    )


def test_vaastav_builder_maps_id_and_code() -> None:
    """Vaastav's id/code become element/player_code."""
    out = build_player_season_from_vaastav(_vaastav_frame(), "2023-24")

    assert out.columns == PLAYER_SEASON.columns
    assert out["element"].to_list() == [1, 2]
    assert out["player_code"].to_list() == [111, 222]
    assert out["season"].to_list() == ["2023-24", "2023-24"]


def test_vaastav_builder_maps_element_type_to_position() -> None:
    """element_type 3 is a midfielder."""
    out = build_player_season_from_vaastav(_vaastav_frame(), "2023-24")

    assert out["position"].to_list() == ["MID", "MID"]


def test_vaastav_builder_nulls_absent_bio_columns() -> None:
    """Seasons whose file predates birth_date get nulls, not an error."""
    out = build_player_season_from_vaastav(_vaastav_frame(), "2023-24")

    assert out["birth_date"].to_list() == [None, None]
    assert out["region"].to_list() == [None, None]
    assert out["team_join_date"].to_list() == [None, None]


def test_vaastav_builder_reads_birth_date_when_present() -> None:
    """From 2024-25 the file carries birth_date; it is parsed as a date."""
    frame = _vaastav_frame().with_columns(
        pl.Series("birth_date", ["1992-06-15", "2001-09-05"])
    )
    out = build_player_season_from_vaastav(frame, "2024-25")

    assert out["birth_date"].to_list() == [
        date(1992, 6, 15),
        date(2001, 9, 5),
    ]


def test_fci_builder_maps_player_id_and_code() -> None:
    """FCI's player_id/player_code become element/player_code."""
    out = build_player_season_from_fci(_fci_frame(), "2025-26")

    assert out.columns == PLAYER_SEASON.columns
    assert out["element"].to_list() == [1, 2]
    assert out["player_code"].to_list() == [111, 222]


def test_fci_builder_normalises_position_labels() -> None:
    """FCI long-form position labels collapse to the canonical short codes."""
    out = build_player_season_from_fci(_fci_frame(), "2025-26")

    assert out["position"].to_list() == ["MID", "GK"]


def test_vaastav_builder_rejects_duplicate_element() -> None:
    """Two rows sharing an element id within a season is a malformed source."""
    frame = _vaastav_frame().with_columns(pl.Series("id", [1, 1]))

    with pytest.raises(ValueError, match="duplicate element"):
        build_player_season_from_vaastav(frame, "2023-24")


def test_vaastav_builder_rejects_duplicate_player_code() -> None:
    """Two rows sharing a player_code within a season is a malformed source."""
    frame = _vaastav_frame().with_columns(pl.Series("code", [111, 111]))

    with pytest.raises(ValueError, match="duplicate player_code"):
        build_player_season_from_vaastav(frame, "2023-24")


def test_fci_builder_rejects_duplicate_element() -> None:
    """The same integrity check applies to the FCI source."""
    frame = _fci_frame().with_columns(pl.Series("player_id", [1, 1]))

    with pytest.raises(ValueError, match="duplicate element"):
        build_player_season_from_fci(frame, "2025-26")


from pathlib import Path
from unittest.mock import MagicMock

from fantasy_football.extraction.player_identity import (
    load_player_identity_data,
)
from fantasy_football.storage.database import get_connection


def _fake_vaastav() -> MagicMock:
    extractor = MagicMock()
    extractor.read_players_raw.side_effect = lambda season: _vaastav_frame()
    return extractor


def _fake_fci() -> MagicMock:
    extractor = MagicMock()
    extractor.read_players.side_effect = lambda long_season: _fci_frame()
    return extractor


def _fake_fplcache(birth: date | None = date(1992, 6, 15)) -> MagicMock:
    extractor = MagicMock()
    extractor.build_player_bio.side_effect = lambda season: pl.DataFrame(
        {
            "element": [1],
            "birth_date": [birth],
            "region": [None],
            "team_join_date": [date(2017, 7, 1)],
        },
        schema_overrides={
            "element": pl.Int64,
            "birth_date": pl.Date,
            "region": pl.Int64,
            "team_join_date": pl.Date,
        },
    )
    return extractor


def test_load_player_identity_data_routes_sources_by_season(
    tmp_path: Path,
) -> None:
    """Seasons up to Vaastav's last come from Vaastav, later ones from FCI."""
    vaastav, fci = _fake_vaastav(), _fake_fci()
    connection = get_connection(tmp_path / "test.duckdb")
    try:
        load_player_identity_data(
            connection,
            current_season="2025-26",
            vaastav=vaastav,
            fci=fci,
            fplcache=_fake_fplcache(),
        )
        out = PLAYER_SEASON.load(connection)
    finally:
        connection.close()

    assert "2025-26" in set(out["season"].to_list())
    assert "2024-25" in set(out["season"].to_list())
    fci.read_players.assert_called_once_with("2025-2026")
    assert "2025-26" not in [
        call.args[0] for call in vaastav.read_players_raw.call_args_list
    ]


def test_load_player_identity_data_enriches_from_fplcache(
    tmp_path: Path,
) -> None:
    """Bio columns fplcache supplies land on the matching element."""
    connection = get_connection(tmp_path / "test.duckdb")
    try:
        load_player_identity_data(
            connection,
            current_season="2025-26",
            vaastav=_fake_vaastav(),
            fci=_fake_fci(),
            fplcache=_fake_fplcache(),
        )
        out = PLAYER_SEASON.load(connection).filter(
            (pl.col("season") == "2025-26") & (pl.col("element") == 1)
        )
    finally:
        connection.close()

    assert out["birth_date"].to_list() == [date(1992, 6, 15)]
    assert out["team_join_date"].to_list() == [date(2017, 7, 1)]


def test_load_player_identity_data_skips_present_immutable_seasons(
    tmp_path: Path,
) -> None:
    """A completed season already stored is not re-fetched on the next run."""
    vaastav, fci = _fake_vaastav(), _fake_fci()
    connection = get_connection(tmp_path / "test.duckdb")
    try:
        load_player_identity_data(
            connection,
            current_season="2025-26",
            vaastav=vaastav,
            fci=fci,
            fplcache=_fake_fplcache(),
        )
        first_count = vaastav.read_players_raw.call_count
        load_player_identity_data(
            connection,
            current_season="2025-26",
            vaastav=vaastav,
            fci=fci,
            fplcache=_fake_fplcache(),
        )
    finally:
        connection.close()

    assert vaastav.read_players_raw.call_count == first_count


def test_load_player_identity_data_survives_a_missing_source_file(
    tmp_path: Path,
) -> None:
    """A 404 on one season is logged and skipped, not fatal."""
    vaastav = MagicMock()

    def _read(season: str) -> pl.DataFrame:
        if season == "2020-21":
            raise requests.HTTPError("404")
        return _vaastav_frame()

    vaastav.read_players_raw.side_effect = _read

    connection = get_connection(tmp_path / "test.duckdb")
    try:
        load_player_identity_data(
            connection,
            current_season="2025-26",
            vaastav=vaastav,
            fci=_fake_fci(),
            fplcache=_fake_fplcache(),
        )
        out = PLAYER_SEASON.load(connection)
    finally:
        connection.close()

    seasons = set(out["season"].to_list())
    assert "2020-21" not in seasons
    assert "2025-26" in seasons


def test_load_player_identity_data_raises_when_current_season_fails(
    tmp_path: Path,
) -> None:
    """A fetch failure for the current season is fatal, not logged and skipped.

    Swallowing it would leave player_season empty for the current season,
    silently reverting every player to a null-history newcomer -- the exact
    failure mode this table exists to prevent.
    """
    fci = MagicMock()
    fci.read_players.side_effect = requests.HTTPError("boom")

    connection = get_connection(tmp_path / "test.duckdb")
    try:
        with pytest.raises(RuntimeError, match="2025-26"):
            load_player_identity_data(
                connection,
                current_season="2025-26",
                vaastav=_fake_vaastav(),
                fci=fci,
                fplcache=_fake_fplcache(),
            )
        out = PLAYER_SEASON.load(connection)
    finally:
        connection.close()

    assert "2025-26" not in set(out["season"].to_list())
