"""Live checks that the identity join survives the Vaastav -> FCI seam.

Marked integration: these hit GitHub and are deselected by default. Run with
``uv run pytest -m integration``.
"""

import pytest

from fantasy_football.extraction.extractor import DataExtractor
from fantasy_football.extraction.fci import FciExtractor
from fantasy_football.extraction.player_identity import (
    build_player_season_from_fci,
    build_player_season_from_vaastav,
)


@pytest.mark.integration
def test_codes_carry_across_the_source_change() -> None:
    """The 2024-25 (Vaastav) and 2025-26 (FCI) seasons share most players.

    Measured at 534 on 2026-07-30. A floor of 500 catches an upstream schema
    change or a silently-renamed column without being brittle to real squad
    turnover.
    """
    vaastav = build_player_season_from_vaastav(
        DataExtractor().read_players_raw("2024-25"), "2024-25"
    )
    fci = build_player_season_from_fci(
        FciExtractor().read_players("2025-2026"), "2025-26"
    )

    overlap = set(vaastav["player_code"].to_list()) & set(
        fci["player_code"].to_list()
    )

    assert len(overlap) >= 500


@pytest.mark.integration
def test_every_season_resolves_a_code_for_every_player() -> None:
    """No source season yields null player_codes."""
    frame = build_player_season_from_vaastav(
        DataExtractor().read_players_raw("2022-23"), "2022-23"
    )

    assert frame["player_code"].null_count() == 0
    assert frame.height > 700
