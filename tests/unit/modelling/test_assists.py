"""The pooled assists rate head.

The tests that matter most are the ones pinning where the assists target
comes from and who the model is allowed to learn from. Both are silent
if they go wrong: a target reading FCI's stricter count leaves the
residual deducting something else, and a training filter reading
``POSITION`` rather than ``TRAINING_POSITIONS`` quietly turns the pooled
model back into a defenders-only one with almost no assists in it.
"""

import polars as pl
import pytest

from fantasy_football.modelling.assists import (
    ASSISTS_SPEC,
    MINUTES_FLOOR,
    SCORING_MINUTES_FLOOR,
    AssistsRatePredictor,
    assists_count_sql,
)
from fantasy_football.modelling.components import (
    ASSIST_POINTS,
    Component,
)
from fantasy_football.modelling.defcon import DefconRatePredictor
from fantasy_football.modelling.defender import (
    DEFENDER_RESIDUAL_SPEC,
    RESIDUAL_TARGET_SQL,
    DefenderResidualPointsPredictor,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy
from fantasy_football.modelling.goals import (
    GOALS_SPEC,
    GoalsRatePredictor,
    goals_count_sql,
)
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
    assists: tuple[int, int] = (0, 0),
    goals: tuple[int, int] = (0, 0),
    total_points: int = 6,
) -> None:
    """Seed one player's two-gameweek run for the given position.

    The Opta rows carry a deliberately different assist count from the
    FPL ones, which is what lets the source test mean anything.
    """
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
                    "xa": 0.2,
                    "chances_created": 2,
                    "accurate_crosses": 1,
                    "assists": assists[gw - 1] + 5,
                    "goals": goals[gw - 1],
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
                    "assists": assists[gw - 1],
                    "goals_scored": goals[gw - 1],
                    # A published nil, not an absent count: the residual
                    # needs one to keep the leg in its fit.
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


def test_target_reads_fpls_count_and_not_the_stricter_feed(
    connection,
) -> None:
    """FPL settles the assist, so FPL's count is the target.

    The two feeds do not measure the same event -- FPL pays for a
    penalty won and a rebound, FCI does not -- so unlike the goals head
    there is no fallback here. The seed gives the two sources
    deliberately different counts, and FCI's must not reach the target.
    """
    _seed_fixtures(connection)
    _seed_player(connection, assists=(1, 2))
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["assists_per_90"].item() == pytest.approx(2.0)
    assert frame["assists"].item() == 2


def test_a_leg_the_fpl_join_misses_has_no_target(connection) -> None:
    """Null only ever means the join missed, and that row cannot be fit."""
    _seed_fixtures(connection)
    _seed_player(connection, assists=(1, 2))
    connection.execute("DELETE FROM player_match_fpl")
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    assert predictor.build_training_data().is_empty()


def test_the_target_is_per_ninety_not_per_match(connection) -> None:
    """Half a match with an assist in it is twice the rate."""
    _seed_fixtures(connection)
    _seed_player(connection, minutes=45, assists=(1, 1))
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["assists_per_90"].item() == pytest.approx(2.0)


# --- Who the model learns from ----------------------------------------


def test_every_outfield_position_reaches_the_training_frame(
    connection,
) -> None:
    """Pooled means pooled: a defenders-only frame is the failure."""
    _seed_fixtures(connection)
    for element, position in ((1, "DEF"), (2, "MID"), (3, "FWD")):
        _seed_player(connection, element=element, position=position)
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame.height == 3


def test_goalkeepers_are_left_out_of_the_fit(connection) -> None:
    """Their assists are rare enough to be only a drag on the mean."""
    _seed_fixtures(connection)
    _seed_player(connection, element=1, position="DEF")
    _seed_player(connection, element=2, position="GK")
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame.height == 1


def test_position_reaches_the_model_as_indicator_columns(
    connection,
) -> None:
    """A tree cannot split on a string, and there is no encoder."""
    _seed_fixtures(connection)
    for element, position in ((1, "DEF"), (2, "MID"), (3, "FWD")):
        _seed_player(connection, element=element, position=position)
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.build_training_data().sort("element")

    assert frame["is_defender"].to_list() == [1.0, 0.0, 0.0]
    assert frame["is_midfielder"].to_list() == [0.0, 1.0, 0.0]
    assert frame["is_forward"].to_list() == [0.0, 0.0, 1.0]


# --- Which rows are dropped -------------------------------------------


def test_short_appearances_are_dropped_from_the_fit(connection) -> None:
    """An assist in a cameo is a per-90 rate with enormous leverage."""
    _seed_fixtures(connection)
    _seed_player(connection, minutes=MINUTES_FLOOR - 1, assists=(1, 1))
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    assert predictor.build_training_data().is_empty()


def test_rows_with_no_form_behind_them_are_dropped_from_the_fit(
    connection,
) -> None:
    """A first appearance carries no signal, only an imputed median."""
    _seed_fixtures(connection)
    _seed_player(connection)
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.build_training_data()

    # Two gameweeks were seeded and only the second has a window.
    assert frame["gw"].to_list() == [2]


# --- The features it reads --------------------------------------------


def test_the_assist_rate_feature_reaches_the_frame(connection) -> None:
    """The head's own history, windowed from the same source as the target."""
    _seed_fixtures(connection)
    _seed_player(connection, assists=(1, 0))
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.build_training_data()

    # One assist in the ninety minutes of the window behind gameweek two.
    assert frame["assists_per90_rolling_5"].item() == pytest.approx(1.0)


def test_creation_features_reach_the_frame(connection) -> None:
    """Volume alongside xA, which is volume weighted by quality."""
    _seed_fixtures(connection)
    _seed_player(connection)
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.build_training_data()

    assert frame["chances_created_per90_rolling_5"].item() == pytest.approx(
        2.0
    )
    assert frame["accurate_crosses_per90_rolling_5"].item() == pytest.approx(
        1.0
    )


def test_penalty_exposure_is_not_an_assists_feature() -> None:
    """FPL credits the assist to whoever won the penalty, not the taker.

    The goals head reads this feature and should; here it has no
    mechanism, and two seasons of data are too few to spend on one.
    """
    from fantasy_football.features.match_form import PENALTY_EXPOSURE_COLUMN

    assert PENALTY_EXPOSURE_COLUMN in GoalsRatePredictor.FEATURES
    assert PENALTY_EXPOSURE_COLUMN not in AssistsRatePredictor.FEATURES


# --- The decomposition holds ------------------------------------------


def test_the_residual_deducts_the_assists_the_head_predicts(
    connection,
) -> None:
    """Otherwise the components sum to more than the total."""
    _seed_fixtures(connection)
    _seed_player(connection, assists=(0, 1), total_points=9)
    predictor = _predictor(
        DefenderResidualPointsPredictor, DEFENDER_RESIDUAL_SPEC, connection
    )

    frame = predictor.build_training_data().filter(pl.col("gw") == 2)

    # 9 points, less 2 for the hour and 3 for the assist.
    assert frame["residual_points_per_90"].item() == pytest.approx(4.0)


def test_the_residual_deducts_both_returns_at_once(connection) -> None:
    """A goal and an assist in one match come off independently."""
    _seed_fixtures(connection)
    _seed_player(connection, assists=(0, 1), goals=(0, 1), total_points=15)
    predictor = _predictor(
        DefenderResidualPointsPredictor, DEFENDER_RESIDUAL_SPEC, connection
    )

    frame = predictor.build_training_data().filter(pl.col("gw") == 2)

    # 15 points, less 2 for the hour, 6 for the goal and 3 for the assist.
    assert frame["residual_points_per_90"].item() == pytest.approx(4.0)


def test_the_residual_deducts_the_expression_the_heads_target(
    connection,
) -> None:
    """One definition, two aliasings, or the components stop summing.

    A second spelling of either count is the failure this guards: the
    target and the deduction would disagree about what a goal or an
    assist is, and nothing downstream would complain.
    """
    assists = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)
    goals = _predictor(GoalsRatePredictor, GOALS_SPEC, connection)

    assert assists_count_sql("pmf") in RESIDUAL_TARGET_SQL
    assert goals_count_sql("pmf", "oc") in RESIDUAL_TARGET_SQL
    assert assists_count_sql() in assists.target_sql
    assert goals_count_sql() in goals.target_sql


def test_a_leg_with_no_assist_count_is_kept_out_of_the_residual_fit(
    connection,
) -> None:
    """A missed join is a deduction that did not happen.

    The count is coalesced to zero so the target stays defined, which
    means the assist points are still inside it. Fitting on such a leg
    would teach the residual to pay for assists the assists component is
    separately paying for.
    """
    _seed_fixtures(connection)
    _seed_player(connection, assists=(0, 1), total_points=9)
    connection.execute("DELETE FROM player_match_fpl")
    predictor = _predictor(
        DefenderResidualPointsPredictor, DEFENDER_RESIDUAL_SPEC, connection
    )

    assert predictor.build_training_data().is_empty()


def test_no_decomposed_def_model_takes_a_minutes_feature() -> None:
    """Minutes are applied once, at composition, and never as a feature."""
    from fantasy_football.modelling.components import MINUTES_OUTPUTS

    for predictor in (
        AssistsRatePredictor,
        GoalsRatePredictor,
        DefconRatePredictor,
        DefenderResidualPointsPredictor,
    ):
        assert not set(predictor.FEATURES) & set(MINUTES_OUTPUTS)
        assert not set(predictor.MINUTES_COLUMNS)


def test_an_assist_pays_the_same_whatever_the_shirt() -> None:
    """Flat across every position, which is why it is not a table."""
    assert ASSIST_POINTS == 3.0
    assert AssistsRatePredictor.COMPONENT is Component.ASSISTS
    assert AssistsRatePredictor.COMPONENT_IMPL.points_per_event == 3.0


# --- Training scope is not scoring scope ------------------------------


def test_every_served_position_is_scored(connection) -> None:
    """One artefact writes rows for all three outfield positions.

    The rows must carry the position they came from, not the model's
    primary one -- nothing downstream can tell the difference, since the
    component name is legitimate either way.
    """
    _seed_fixtures(connection)
    for element, position in ((1, "DEF"), (2, "MID"), (3, "FWD")):
        _seed_player(connection, element=element, position=position)
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

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
    fit here would delete predictions the other four were happy to make.
    """
    _seed_fixtures(connection)
    _seed_player(connection, minutes=MINUTES_FLOOR - 1, assists=(1, 1))
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    assert predictor.build_training_data().is_empty()
    assert not predictor.scoring_frame().is_empty()


def test_the_scoring_floor_matches_the_sibling_components() -> None:
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
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    assert predictor.build_training_data()["gw"].to_list() == [2]
    assert predictor.scoring_frame()["gw"].sort().to_list() == [1, 2]


def test_a_scored_debutant_is_flagged_rather_than_imputed_silently(
    connection,
) -> None:
    """The flag survives to the model rather than arriving as a null."""
    _seed_fixtures(connection)
    _seed_player(connection)
    predictor = _predictor(AssistsRatePredictor, ASSISTS_SPEC, connection)

    frame = predictor.scoring_frame().sort("gw")

    assert frame["has_no_form"].to_list() == [1.0, 0.0]
