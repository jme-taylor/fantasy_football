"""Live checks that the match-level loaders work against the real sources.

Marked ``integration``: these hit GitHub and are deselected by default.
Run with ``uv run pytest -m integration``.
"""

import pytest

from fantasy_football.extraction.player_match_fpl import VaastavMatchLoader
from fantasy_football.extraction.player_match_opta import FciMatchStatsLoader
from fantasy_football.storage.coverage import FCI_SEASONS, VAASTAV_SEASONS
from fantasy_football.storage.tables import (
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
)


@pytest.mark.integration
def test_vaastav_available_seasons_matches_the_coverage_map():
    """The seasons the repo publishes are the ones the coverage map claims."""
    assert tuple(VaastavMatchLoader().available_seasons()) == VAASTAV_SEASONS


@pytest.mark.integration
def test_fci_available_seasons_matches_the_coverage_map():
    """The FCI seasons on disk are the ones the coverage map claims."""
    assert tuple(FciMatchStatsLoader().available_seasons()) == FCI_SEASONS


@pytest.mark.integration
def test_a_legacy_vaastav_season_loads_and_conforms():
    """2016-17 is non-UTF-8 and column-sparse; it still reduces to schema."""
    frame = VaastavMatchLoader().load_season("2016-17")
    assert frame.columns == PLAYER_MATCH_FPL.columns
    assert frame.height > 20000
    # Not published until 2022-23.
    assert frame["expected_goals"].null_count() == frame.height
    # Published in every season.
    assert frame["total_points"].null_count() == 0


@pytest.mark.integration
def test_a_recent_vaastav_season_carries_the_expected_stats():
    """2025-26 has expected_goals and defensive_contribution populated."""
    frame = VaastavMatchLoader().load_season("2025-26")
    assert frame["expected_goals"].null_count() < frame.height
    assert frame["defensive_contribution"].null_count() < frame.height


@pytest.mark.integration
def test_an_fci_season_loads_with_every_competition():
    """FCI 2025-26 carries European and cup rows alongside the league."""
    frame = FciMatchStatsLoader().load_season("2025-26")
    assert frame.columns == PLAYER_MATCH_OPTA.columns
    competitions = set(frame["competition"].unique().to_list())
    assert "prem" in competitions
    assert len(competitions) > 1


@pytest.mark.integration
def test_vaastav_primary_key_holds_on_real_data():
    """The (season, gw, element, fixture) key is unique in a real season."""
    frame = VaastavMatchLoader().load_season("2025-26")
    key = ["season", "gw", "element", "fixture"]
    assert frame.select(key).unique().height == frame.height
