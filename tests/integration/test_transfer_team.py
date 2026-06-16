"""Integration: real fplcache data gives transferred players their real club.

Hits the live Randdalf/fplcache and FPL APIs. Deselected by default; run with
``uv run pytest -m integration``.
"""

import pytest

from fantasy_football.extraction.fplcache import FplCacheExtractor

# FPL element ids and pre-transfer team_codes for the 2025-26 season.
SEMENYO_ELEMENT = 82
GUEHI_ELEMENT = 260
BOURNEMOUTH_CODE = 91
CRYSTAL_PALACE_CODE = 31
MAN_CITY_CODE = 43


@pytest.mark.integration
def test_gw1_team_is_pre_transfer_club() -> None:
    """In GW1, Semenyo/Guehi are at their original clubs, not Man City."""
    extractor = FplCacheExtractor()
    table = extractor.build_player_gw_team([1])

    semenyo = table.filter(
        (table["gw"] == 1) & (table["element"] == SEMENYO_ELEMENT)
    )
    guehi = table.filter(
        (table["gw"] == 1) & (table["element"] == GUEHI_ELEMENT)
    )

    assert semenyo.height == 1
    assert guehi.height == 1
    semenyo_code = semenyo["team_code"][0]
    guehi_code = guehi["team_code"][0]
    assert semenyo_code != MAN_CITY_CODE
    assert guehi_code != MAN_CITY_CODE
    assert semenyo_code == BOURNEMOUTH_CODE
    assert guehi_code == CRYSTAL_PALACE_CODE
