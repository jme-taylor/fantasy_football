"""The CBIRT rate head, which serves midfielders and forwards.

The tests that matter most are the ones pinning where the count comes
from. Three sources feed it and they cover different seasons, so a
mistake in the order is invisible until a season with only the third
one -- which is every season the live pipeline scores.
"""

import pytest

from fantasy_football.constants import DEFCON_THRESHOLD_BY_POSITION
from fantasy_football.modelling.components import MINUTES_OUTPUTS, Component
from fantasy_football.modelling.defcon import (
    CBIRT_SPEC,
    MINUTES_FLOOR,
    CbirtRatePredictor,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy
from fantasy_football.storage.tables import (
    PLAYER_MATCH,
    PLAYER_MATCH_FPL,
    PLAYER_MATCH_OPTA,
    PLAYER_SEASON,
    PLAYER_WEEK,
    TEAM_FIXTURE,
)
from tests.unit.modelling.conftest import (
    ARSENAL,
    GW1_KICKOFF,
    SEASON,
    SLUGS,
    TEAM_IDS,
    UNITED,
    append_rows,
)


def _predictor(connection) -> CbirtRatePredictor:
    """Build the CBIRT head against the test db."""
    return CbirtRatePredictor(
        experiment_name="test",
        params={},
        model_spec=CBIRT_SPEC,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def _seed_one_match(
    connection,
    *,
    position: str = "MID",
    element: int = 1,
    minutes: int = 90,
    opta_counters: dict[str, int] | None = None,
    opta_defcon: int | None = None,
    fpl_defcon: int | None = None,
) -> None:
    """Seed a single played fixture with every CBIRT source."""
    match_id = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[ARSENAL]}"
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": element, "position": position}],
    )
    append_rows(
        PLAYER_WEEK,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": element,
                "position": position,
                "team": UNITED,
            }
        ],
    )
    append_rows(
        PLAYER_MATCH,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": element,
                "opponent": TEAM_IDS[ARSENAL],
                "is_home": True,
                "minutes": minutes,
                "total_points": 2,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": GW1_KICKOFF,
            }
        ],
    )
    append_rows(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "team": team,
                "is_home": is_home,
                "opposition": opposition,
                "kickoff_time": GW1_KICKOFF,
            }
            for team, is_home, opposition in [
                (UNITED, True, ARSENAL),
                (ARSENAL, False, UNITED),
            ]
        ],
    )
    if opta_counters is not None or opta_defcon is not None:
        append_rows(
            PLAYER_MATCH_OPTA,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": 1,
                    "element": element,
                    "match_id": match_id,
                    "competition": "prem",
                    "minutes_played": minutes,
                    "defensive_contributions": opta_defcon,
                    **(opta_counters or {}),
                }
            ],
        )
    append_rows(
        PLAYER_MATCH_FPL,
        connection,
        [
            {
                "season": SEASON,
                "gw": 1,
                "element": element,
                "fixture": 1,
                "opponent_team": TEAM_IDS[ARSENAL],
                "minutes": minutes,
                "defensive_contribution": fpl_defcon,
            }
        ],
    )


# --- Where the CBIRT target comes from --------------------------------


def test_target_prefers_fpls_own_published_count(connection) -> None:
    """FPL's count wins wherever it exists.

    It is the count the rule was actually scored on, so it is the target
    for every season that publishes it.
    """
    _seed_one_match(
        connection,
        opta_counters={"tackles": 1, "recoveries": 1},
        opta_defcon=2,
        fpl_defcon=14,
    )

    frame = _predictor(connection).build_training_data()

    assert frame["cbirt_count"].item() == 14
    assert frame["cbirt_per_90"].item() == pytest.approx(14.0)


def test_target_prefers_fcis_published_count_to_its_counters(
    connection,
) -> None:
    """FCI's own count beats summing FCI's counters by hand.

    Reconciled on 2025-26 it agrees with FPL on every row it covers,
    where the hand-summed reconstruction agrees on 93%. Vaastav's column
    ends at 2025-26, so this is what the live season falls back to.
    """
    _seed_one_match(
        connection,
        opta_counters={"tackles": 3, "recoveries": 2},
        opta_defcon=13,
        fpl_defcon=None,
    )

    frame = _predictor(connection).build_training_data()

    assert frame["cbirt_count"].item() == 13


def test_target_falls_back_to_the_five_counter_reconstruction(
    connection,
) -> None:
    """With neither published count, the five counters are summed."""
    _seed_one_match(
        connection,
        opta_counters={
            "clearances": 2,
            "blocks": 1,
            "interceptions": 3,
            "tackles": 4,
            "recoveries": 6,
        },
    )

    frame = _predictor(connection).build_training_data()

    assert frame["cbirt_count"].item() == 16


def test_recoveries_are_counted_where_the_defender_head_drops_them(
    connection,
) -> None:
    """Recoveries are the whole difference between the two counts."""
    _seed_one_match(
        connection,
        opta_counters={"tackles": 4, "recoveries": 6},
    )

    frame = _predictor(connection).build_training_data()

    assert frame["cbirt_count"].item() == 10


def test_a_row_with_no_count_anywhere_is_dropped(connection) -> None:
    """An unobserved player did not make no tackles."""
    _seed_one_match(connection)

    frame = _predictor(connection).build_training_data()

    assert frame.is_empty()


def test_short_appearances_are_dropped(connection) -> None:
    """A per-90 rate off a cameo is arithmetic noise."""
    _seed_one_match(
        connection,
        minutes=MINUTES_FLOOR - 1,
        opta_counters={"tackles": 2, "recoveries": 1},
    )

    frame = _predictor(connection).build_training_data()

    assert frame.is_empty()


# --- What the head is, structurally -----------------------------------


def test_the_head_reads_no_minutes_feature(connection) -> None:
    """Minutes are applied once, at composition, not twice."""
    assert not set(CbirtRatePredictor.FEATURES) & set(MINUTES_OUTPUTS)
    assert CbirtRatePredictor.MINUTES_COLUMNS == []


def test_the_head_is_scored_against_twelve_not_ten() -> None:
    """Midfielders and forwards are paid at a different threshold."""
    assert (
        CbirtRatePredictor.COMPONENT_IMPL.threshold
        == (DEFCON_THRESHOLD_BY_POSITION["MID"])
    )
    assert CbirtRatePredictor.COMPONENT_IMPL.threshold == 12


def test_the_head_pools_midfielders_and_forwards() -> None:
    """One head, both positions, with a dummy to tell them apart."""
    predictor_positions = set(CbirtRatePredictor.TRAINING_POSITIONS)

    assert predictor_positions == {"MID", "FWD"}
    assert "is_midfielder" in CbirtRatePredictor.FEATURES
    assert "is_forward" in CbirtRatePredictor.FEATURES


def test_the_head_writes_the_defcon_component() -> None:
    """It replaces the same points the defender head does."""
    assert CBIRT_SPEC.component == Component.DEFCON
    assert CBIRT_SPEC.registered_model_name == "cbirt_rate_regressor"


def test_forwards_are_in_the_training_frame(connection) -> None:
    """A pooled head that silently dropped forwards would still fit."""
    _seed_one_match(
        connection,
        position="FWD",
        opta_counters={"tackles": 5, "recoveries": 8},
    )

    frame = _predictor(connection).build_training_data()

    assert frame["cbirt_count"].item() == 13
    assert frame["is_forward"].item() == pytest.approx(1.0)
    assert frame["is_midfielder"].item() == pytest.approx(0.0)


def test_the_fold_metrics_score_at_match_scale(connection) -> None:
    """Scored on P(count >= 12), not on the per-90 rate."""
    _seed_one_match(
        connection,
        opta_counters={"tackles": 8, "recoveries": 8},
    )
    predictor = _predictor(connection)
    frame = predictor.build_training_data()

    metrics = predictor.fold_metrics(frame, [16.0])

    assert metrics.hit_rate == pytest.approx(1.0)
    assert 0.0 <= metrics.brier <= 1.0
