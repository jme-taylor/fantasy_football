"""The defensive-contribution rate model and the residual points model.

Both are the decomposed halves of the DEF prediction, and both are
minutes-free by construction. The tests that matter most here are the
ones pinning where the CBIT target comes from and what the residual
target deducts, since a mistake in either is invisible downstream.
"""

import polars as pl
import pytest

from fantasy_football.modelling.components import Component
from fantasy_football.modelling.defcon import (
    CBIRT_SPEC,
    DEFCON_SPEC,
    MINUTES_FLOOR,
    CbirtRatePredictor,
    DefconRatePredictor,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    MINUTES_PREDICTION,
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

PRIOR_SEASON = "2024-25"


def _predictor(cls, spec, connection):
    """Build one of the decomposed DEF models against the test db."""
    return cls(
        experiment_name="test",
        params={},
        model_spec=spec,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def _seed_one_match(
    connection,
    *,
    season: str = SEASON,
    minutes: int = 90,
    total_points: int = 6,
    opta_counters: dict[str, int] | None = None,
    fpl_defcon: int | None = None,
) -> None:
    """Seed a single played DEF fixture with both CBIT sources."""
    match_id = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[ARSENAL]}"
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": season, "element": 1, "position": "DEF"}],
    )
    append_rows(
        PLAYER_WEEK,
        connection,
        [
            {
                "season": season,
                "gw": 1,
                "element": 1,
                "position": "DEF",
                "team": UNITED,
            }
        ],
    )
    append_rows(
        PLAYER_MATCH,
        connection,
        [
            {
                "season": season,
                "gw": 1,
                "element": 1,
                "opponent": TEAM_IDS[ARSENAL],
                "is_home": True,
                "minutes": minutes,
                "total_points": total_points,
                "yellow_cards": 0,
                "red_cards": 0,
                "kickoff_time": GW1_KICKOFF,
            }
        ],
    )
    # opta_match resolves the club-name bridge through fpl_team_id,
    # which is built from the fixture list.
    append_rows(
        TEAM_FIXTURE,
        connection,
        [
            {
                "season": season,
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
    if opta_counters is not None:
        append_rows(
            PLAYER_MATCH_OPTA,
            connection,
            [
                {
                    "season": season,
                    "gw": 1,
                    "element": 1,
                    "match_id": match_id,
                    "competition": "prem",
                    "minutes_played": minutes,
                    **opta_counters,
                }
            ],
        )
    append_rows(
        PLAYER_MATCH_FPL,
        connection,
        [
            {
                "season": season,
                "gw": 1,
                "element": 1,
                "fixture": 1,
                "opponent_team": TEAM_IDS[ARSENAL],
                "minutes": minutes,
                "defensive_contribution": fpl_defcon,
                # Published nils, not absent counts. The residual
                # excludes legs with no published goal, assist or clean
                # sheet, since those points are still inside its target.
                "goals_scored": 0,
                "assists": 0,
                "clean_sheets": 0,
            }
        ],
    )


# --- Where the CBIT target comes from ---------------------------------


def test_target_prefers_fpls_own_published_count(connection) -> None:
    """FPL's count wins wherever it exists.

    The two providers disagree by one on a few per cent of defender
    gameweeks. FPL's is the count the rule was actually scored on, so it
    is the target for every season that publishes it.
    """
    _seed_one_match(
        connection,
        opta_counters={
            "clearances": 1,
            "blocks": 1,
            "interceptions": 1,
            "tackles": 1,
        },
        fpl_defcon=12,
    )
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["cbit_per_90"].item() == pytest.approx(12.0)
    assert frame["cbit_count"].item() == 12


def test_target_falls_back_to_the_opta_reconstruction(connection) -> None:
    """Where FPL publishes nothing, the four counters are summed."""
    _seed_one_match(
        connection,
        opta_counters={
            "clearances": 4,
            "blocks": 1,
            "interceptions": 2,
            "tackles": 3,
        },
        fpl_defcon=None,
    )
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["cbit_per_90"].item() == pytest.approx(10.0)


def test_recoveries_are_not_part_of_the_defender_count(connection) -> None:
    """Recoveries belong to the midfield threshold, not this one."""
    _seed_one_match(
        connection,
        opta_counters={
            "clearances": 1,
            "blocks": 1,
            "interceptions": 1,
            "tackles": 1,
            "recoveries": 20,
        },
        fpl_defcon=None,
    )
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["cbit_per_90"].item() == pytest.approx(4.0)


def test_the_rate_is_per_ninety_not_per_match(connection) -> None:
    """Half a match of counters is twice the rate."""
    _seed_one_match(
        connection,
        minutes=45,
        opta_counters={"tackles": 5},
        fpl_defcon=None,
    )
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["cbit_per_90"].item() == pytest.approx(10.0)


def test_rows_with_no_count_at_all_are_dropped(connection) -> None:
    """A season neither source covers cannot train a rate."""
    _seed_one_match(connection, opta_counters=None, fpl_defcon=None)
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)

    assert predictor.build_training_data().is_empty()


def test_short_appearances_are_dropped(connection) -> None:
    """A per-90 rate off a few minutes is arithmetic noise."""
    _seed_one_match(
        connection,
        minutes=MINUTES_FLOOR - 1,
        opta_counters={"tackles": 2},
        fpl_defcon=None,
    )
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)

    assert predictor.build_training_data().is_empty()


def test_seasons_before_any_cbit_source_are_excluded(connection) -> None:
    """2016-19 carries counters but is a different era, and is left out."""
    assert "2018-19" not in DefconRatePredictor.TRAINING_SEASONS
    assert "2024-25" in DefconRatePredictor.TRAINING_SEASONS


# --- Metrics ---------------------------------------------------------


def test_fold_metrics_score_the_threshold_not_the_rate(
    connection, synthetic_frame
) -> None:
    """Scoring happens at match scale, against the base rate."""
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)
    frame = synthetic_frame(predictor, n_gws=2, n_players=6)
    frame = frame.with_columns(
        cbit_count=pl.when(pl.col("element") > 3).then(12).otherwise(2)
    )

    metrics = predictor.fold_metrics(frame, [8.0] * frame.height).as_dict()

    assert 0.0 <= metrics["brier"] <= 1.0
    assert metrics["hit_rate"] == pytest.approx(0.5)
    assert "rate_mae" in metrics


def test_a_useless_head_scores_no_skill(connection, synthetic_frame) -> None:
    """Predicting the base rate everywhere scores zero, not well.

    Defcon fires for roughly a quarter of defenders, so a head with no
    signal would look accurate on raw error alone.
    """
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)
    frame = synthetic_frame(predictor, n_gws=2, n_players=6).with_columns(
        cbit_count=pl.when(pl.col("element") > 3).then(12).otherwise(2)
    )

    metrics = predictor.fold_metrics(frame, [0.0] * frame.height)

    assert metrics.skill_score <= 0.0


# --- The residual target ---------------------------------------------


# --- Weighting -------------------------------------------------------


# --- Components produced ---------------------------------------------


def test_defcon_rows_are_stored_as_defcon_components(
    connection, synthetic_frame
) -> None:
    """The rate model writes defcon component rows, not points rows."""
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)
    frame = synthetic_frame(predictor, n_gws=2, n_players=4)
    append_rows(
        MINUTES_PREDICTION,
        connection,
        [
            {
                "season": row["season"],
                "gw": row["gw"],
                "element": row["element"],
                "opponent": row["opponent"],
                "p_zero": 0.1,
                "p_partial": 0.2,
                "p_sixty_plus": 0.7,
                "expected_minutes": 80.0,
                "model_version": "1",
                "prediction_kind": BACKFILL_KIND,
            }
            for row in frame.iter_rows(named=True)
        ],
    )
    model = predictor.train_final(frame)

    rows = predictor.build_prediction_rows(frame, model, "4", BACKFILL_KIND)

    assert rows["component"].unique().to_list() == [str(Component.DEFCON)]
    assert rows["points"].max() <= 2.0
    assert rows["points"].min() >= 0.0
    assert rows["diagnostics"].null_count() == 0


def test_fold_metrics_survive_a_fold_where_nobody_clears_the_threshold(
    connection, synthetic_frame
) -> None:
    """A quiet gameweek is a score, not a crash.

    Both sklearn metrics infer their labels from the data unless told, so
    a single-class fold raises. Reachable under the per-gameweek fold
    strategy the standalone training script can use.
    """
    predictor = _predictor(DefconRatePredictor, DEFCON_SPEC, connection)
    frame = synthetic_frame(predictor, n_gws=1, n_players=4).with_columns(
        cbit_count=pl.lit(2)
    )

    metrics = predictor.fold_metrics(frame, [3.0] * frame.height)

    assert metrics.hit_rate == pytest.approx(0.0)
    assert metrics.brier >= 0.0


# --- Weighting -------------------------------------------------------


@pytest.mark.parametrize(
    ("cls", "spec"),
    [
        (DefconRatePredictor, DEFCON_SPEC),
        (CbirtRatePredictor, CBIRT_SPEC),
    ],
    ids=["cbit", "cbirt"],
)
def test_training_rows_are_weighted_by_minutes(
    cls, spec, connection, synthetic_frame
) -> None:
    """A full ninety counts for more than a cameo when fitting."""
    predictor = _predictor(cls, spec, connection)
    frame = synthetic_frame(predictor, n_gws=1, n_players=3).with_columns(
        minutes=pl.Series([90.0, 45.0, 9.0])
    )

    assert predictor.sample_weight(frame) == pytest.approx([90.0, 45.0, 9.0])
