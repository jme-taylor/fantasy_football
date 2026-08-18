"""The pooled goals rate head.

The tests that matter most are the ones pinning where the goals target
comes from and who the model is allowed to learn from. Both are silent
if they go wrong: a target reading the wrong source leaves the residual
deducting something else, and a training filter reading ``POSITION``
rather than ``TRAINING_POSITIONS`` quietly turns the pooled model back
into a defenders-only one with almost no goals in it.
"""

import polars as pl
import pytest

from fantasy_football.modelling.components import (
    GOALS_POINTS_BY_POSITION,
    Component,
)
from fantasy_football.modelling.defcon import DefconRatePredictor
from fantasy_football.modelling.defender import (
    DEFENDER_RESIDUAL_SPEC,
    DefenderResidualPointsPredictor,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy
from fantasy_football.modelling.goals import (
    GOALS_SPEC,
    MINUTES_FLOOR,
    SCORING_MINUTES_FLOOR,
    GoalsRatePredictor,
    goals_count_sql,
)
from fantasy_football.modelling.points import position_dummy_names
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

# Two gameweeks: the first builds the form window, the second is the row
# under test. A single-gameweek seed is dropped by the no-form filter.
GWS = (1, 2)


def _predictor(cls, spec, connection):
    """Build a predictor against the test database."""
    return cls(
        experiment_name="test",
        params={},
        model_spec=spec,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def _seed_player(
    connection,
    *,
    element: int = 1,
    position: str = "DEF",
    minutes: int = 90,
    goals: tuple[int, int] = (0, 0),
    total_points: int = 6,
) -> None:
    """Seed one player's two-gameweek run for the given position."""
    match_id = f"25-26-prem-{SLUGS[UNITED]}-vs-{SLUGS[ARSENAL]}"
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": SEASON, "element": element, "position": position}],
    )
    for gw in GWS:
        kickoff = GW1_KICKOFF.replace(day=GW1_KICKOFF.day + 7 * (gw - 1))
        append_rows(
            PLAYER_WEEK,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
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
                    "gw": gw,
                    "element": element,
                    "opponent": TEAM_IDS[ARSENAL],
                    "is_home": True,
                    "minutes": minutes,
                    "total_points": total_points,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "kickoff_time": kickoff,
                }
            ],
        )
        append_rows(
            PLAYER_MATCH_OPTA,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "match_id": match_id,
                    "competition": "prem",
                    "minutes_played": minutes,
                    "xg": 0.3,
                    "goals": goals[gw - 1] + 5,
                }
            ],
        )
        append_rows(
            PLAYER_MATCH_FPL,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "element": element,
                    "fixture": gw,
                    "opponent_team": TEAM_IDS[ARSENAL],
                    "minutes": minutes,
                    "goals_scored": goals[gw - 1],
                    # Published nils, not absent counts: the residual
                    # needs them to keep the leg in its fit.
                    "assists": 0,
                    "clean_sheets": 0,
                }
            ],
        )


def _seed_fixtures(connection) -> None:
    """Seed both clubs' fixtures, which the club-name bridge reads."""
    for gw in GWS:
        kickoff = GW1_KICKOFF.replace(day=GW1_KICKOFF.day + 7 * (gw - 1))
        append_rows(
            TEAM_FIXTURE,
            connection,
            [
                {
                    "season": SEASON,
                    "gw": gw,
                    "team": team,
                    "is_home": is_home,
                    "opposition": opposition,
                    "kickoff_time": kickoff,
                }
                for team, is_home, opposition in [
                    (UNITED, True, ARSENAL),
                    (ARSENAL, False, UNITED),
                ]
            ],
        )


# --- What the model predicts ------------------------------------------


def test_target_prefers_fpls_own_published_count(connection) -> None:
    """FPL's count wins wherever it exists.

    The seed gives the two sources deliberately different counts. FPL's
    is what the points were actually settled on, and it is what the
    residual deducts, so it has to be what the target reads.
    """
    _seed_fixtures(connection)
    _seed_player(connection, goals=(1, 2))
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["goals_per_90"].item() == pytest.approx(2.0)
    assert frame["goals_scored"].item() == 2


def test_target_falls_back_to_the_opta_feed(connection) -> None:
    """Vaastav stops at 2025-26; the live season needs the fallback."""
    _seed_fixtures(connection)
    _seed_player(connection, goals=(0, 0))
    connection.execute("DELETE FROM player_match_fpl")
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    frame = predictor.build_training_data()

    # The Opta seed carries five more goals than the FPL one.
    assert frame["goals_scored"].item() == 5


def test_the_target_is_per_ninety_not_per_match(connection) -> None:
    """Half a match with a goal in it is twice the rate."""
    _seed_fixtures(connection)
    _seed_player(connection, minutes=45, goals=(1, 1))
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["goals_per_90"].item() == pytest.approx(2.0)


# --- Who the model learns from ----------------------------------------


def test_every_outfield_position_reaches_the_training_frame(
    connection,
) -> None:
    """Pooled means pooled: a defenders-only frame is the failure."""
    _seed_fixtures(connection)
    for element, position in ((1, "DEF"), (2, "MID"), (3, "FWD")):
        _seed_player(connection, element=element, position=position)
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame.height == 3


def test_goalkeepers_are_left_out_of_the_fit(connection) -> None:
    """Their goals are so rare the rows are only a drag on the mean."""
    _seed_fixtures(connection)
    _seed_player(connection, element=1, position="DEF")
    _seed_player(connection, element=2, position="GK")
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame.height == 1


def test_position_reaches_the_model_as_indicator_columns(
    connection,
) -> None:
    """A tree cannot split on a string, and there is no encoder."""
    _seed_fixtures(connection)
    for element, position in ((1, "DEF"), (2, "MID"), (3, "FWD")):
        _seed_player(connection, element=element, position=position)
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    frame = predictor.build_training_data().sort("element")

    assert frame["is_defender"].to_list() == [1.0, 0.0, 0.0]
    assert frame["is_midfielder"].to_list() == [0.0, 1.0, 0.0]
    assert frame["is_forward"].to_list() == [0.0, 0.0, 1.0]


def test_a_single_position_model_gets_no_indicator_columns() -> None:
    """Constant columns teach a tree nothing and cost it a split."""
    assert position_dummy_names(("DEF",)) == []
    assert position_dummy_names(("DEF", "MID")) == [
        "is_defender",
        "is_midfielder",
    ]


# --- Which rows are dropped -------------------------------------------


def test_short_appearances_are_dropped_from_the_fit(connection) -> None:
    """A goal in a cameo is a per-90 rate with enormous leverage."""
    _seed_fixtures(connection)
    _seed_player(connection, minutes=MINUTES_FLOOR - 1, goals=(1, 1))
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    assert predictor.build_training_data().is_empty()


def test_rows_with_no_form_behind_them_are_dropped_from_the_fit(
    connection,
) -> None:
    """A first appearance carries no signal, only an imputed median."""
    _seed_fixtures(connection)
    _seed_player(connection)
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    frame = predictor.build_training_data()

    # Two gameweeks were seeded and only the second has a window.
    assert frame["gw"].to_list() == [2]


# --- The decomposition holds ------------------------------------------


def test_the_residual_deducts_the_goals_the_head_predicts(
    connection,
) -> None:
    """Otherwise the components sum to more than the total."""
    _seed_fixtures(connection)
    _seed_player(connection, goals=(0, 1), total_points=12)
    predictor = _predictor(
        DefenderResidualPointsPredictor, DEFENDER_RESIDUAL_SPEC, connection
    )

    frame = predictor.build_training_data().filter(pl.col("gw") == 2)

    # 12 points, less 2 for the hour and 6 for the goal.
    assert frame["residual_points_per_90"].item() == pytest.approx(4.0)


def test_the_residual_deducts_the_expression_the_head_targets(
    connection,
) -> None:
    """One definition, two aliasings.

    ``goals_count_sql`` is a function precisely so the target and the
    deduction cannot be spelled two ways. Until now that was defended by
    a comment, and a second spelling would leave the components summing
    to something other than the total with nothing failing.
    """
    from fantasy_football.modelling.defender import RESIDUAL_TARGET_SQL

    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    assert goals_count_sql("pmf", "oc") in RESIDUAL_TARGET_SQL
    assert goals_count_sql() in predictor.target_sql


def test_no_decomposed_def_model_takes_a_minutes_feature() -> None:
    """Minutes are applied once, at composition, and never as a feature."""
    from fantasy_football.modelling.components import MINUTES_OUTPUTS

    for predictor in (
        GoalsRatePredictor,
        DefconRatePredictor,
        DefenderResidualPointsPredictor,
    ):
        assert not set(predictor.FEATURES) & set(MINUTES_OUTPUTS)
        assert not set(predictor.MINUTES_COLUMNS)


def test_a_goal_is_worth_what_the_position_pays() -> None:
    """One rate, three prices, which is why the model predicts a rate."""
    assert GOALS_POINTS_BY_POSITION["DEF"] == 6.0
    assert GOALS_POINTS_BY_POSITION["MID"] == 5.0
    assert GOALS_POINTS_BY_POSITION["FWD"] == 4.0
    assert GoalsRatePredictor.COMPONENT is Component.GOALS


# --- Training scope is not scoring scope ------------------------------


def test_every_served_position_is_scored(connection) -> None:
    """One artefact writes rows for all three outfield positions.

    The rows must carry the position they came from, not the model's
    primary one: a midfielder filed as a defender has his goals priced
    at six points, and nothing downstream can tell -- the component name
    is legitimate either way.
    """
    _seed_fixtures(connection)
    for element, position in ((1, "DEF"), (2, "MID"), (3, "FWD")):
        _seed_player(connection, element=element, position=position)
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    assert predictor.build_training_data().height == 3
    scored = predictor.scoring_frame()
    assert sorted(scored["element"].unique().to_list()) == [1, 2, 3]

    stamped = (
        scored.with_columns(position=predictor._served_position())
        .unique(subset=["element"])
        .sort("element")
    )
    assert stamped["position"].to_list() == ["DEF", "MID", "FWD"]


def test_short_appearances_are_still_scored(connection) -> None:
    """A leg dropped from the fit still needs its component written.

    DEF declares five components now, and a leg missing any one of them
    is dropped from the composed prediction altogether -- so a stricter
    fit here would delete predictions the other four were happy to
    make.
    """
    _seed_fixtures(connection)
    _seed_player(connection, minutes=MINUTES_FLOOR - 1, goals=(1, 1))
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    assert predictor.build_training_data().is_empty()
    assert not predictor.scoring_frame().is_empty()


def test_the_scoring_floor_matches_the_sibling_components(
    connection,
) -> None:
    """The floor a leg must clear to be scored is the shared one."""
    from fantasy_football.modelling.defcon import (
        MINUTES_FLOOR as DEFCON_FLOOR,
    )

    assert SCORING_MINUTES_FLOOR == DEFCON_FLOOR
    assert MINUTES_FLOOR > SCORING_MINUTES_FLOOR


def test_no_form_rows_are_still_scored(connection) -> None:
    """A debutant has to be predicted for, however little is known."""
    _seed_fixtures(connection)
    _seed_player(connection)
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    assert predictor.build_training_data()["gw"].to_list() == [2]
    assert predictor.scoring_frame()["gw"].sort().to_list() == [1, 2]


def test_a_scored_debutant_is_flagged_rather_than_imputed_silently(
    connection,
) -> None:
    """The flag survives to the model rather than arriving as a null."""
    _seed_fixtures(connection)
    _seed_player(connection)
    predictor = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    frame = predictor.scoring_frame().sort("gw")

    assert frame["has_no_form"].to_list() == [1.0, 0.0]
