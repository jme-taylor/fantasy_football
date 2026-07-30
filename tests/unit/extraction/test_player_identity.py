from datetime import date

import polars as pl
import pytest

from fantasy_football.extraction.player_identity import (
    build_player_season_from_fci,
    build_player_season_from_vaastav,
)
from fantasy_football.storage.database import PLAYER_SEASON_COLUMNS


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

    assert out.columns == PLAYER_SEASON_COLUMNS
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

    assert out.columns == PLAYER_SEASON_COLUMNS
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
