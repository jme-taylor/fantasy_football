"""What makes the defender model a defender model.

Everything the defender shares with the other positions is exercised
once per position in ``test_points.py``. What is left here is the
column semantics: a defender is scored on what his own club concedes
and on what the opposition creates.
"""

import polars as pl
import pytest

from fantasy_football.features.views import register_feature_views
from fantasy_football.modelling.defender import (
    DEFENDER_SPEC,
    POSITION,
    DefenderPointsPredictor,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy


@pytest.fixture
def predictor(connection) -> DefenderPointsPredictor:
    """Return a ``DefenderPointsPredictor`` bound to the test connection."""
    return DefenderPointsPredictor(
        experiment_name="test-defender",
        params={},
        model_spec=DEFENDER_SPEC,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def test_spec_position_matches_the_class() -> None:
    """The spec's position is what scopes the partition rewrite.

    A spec pointing at a different position from the class that stamps
    the rows would delete another position's partition and write its own
    rows into it.
    """
    assert DEFENDER_SPEC.position == POSITION
    assert DefenderPointsPredictor.POSITION == POSITION


def test_own_columns_are_defensive_and_opposition_columns_attacking() -> None:
    """A defender's own record is what his club concedes."""
    assert DefenderPointsPredictor.OWN_TEAM_COLUMNS == [
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
    ]
    assert DefenderPointsPredictor.OPPOSITION_COLUMNS == [
        "xg_for_rolling_5",
        "goals_for_rolling_5",
    ]


def test_features_include_the_minutes_bucket_probabilities() -> None:
    """The defender model reads the full minutes distribution.

    This is the deliberate difference from the forwards model, which
    takes expected_minutes alone -- a clean sheet needs sixty minutes on
    the pitch, so how the minutes are distributed matters here.
    """
    for column in ("p_zero", "p_partial", "p_sixty_plus"):
        assert column in DefenderPointsPredictor.MINUTES_COLUMNS
        assert column in DefenderPointsPredictor.FEATURES


def test_every_computed_column_reaches_the_feature_list() -> None:
    """No form column is computed and then silently dropped."""
    computed = (
        DefenderPointsPredictor.PLAYER_FORM_COLUMNS
        + DefenderPointsPredictor.OWN_TEAM_COLUMNS
        + DefenderPointsPredictor.OPPOSITION_COLUMNS
        + DefenderPointsPredictor.MINUTES_COLUMNS
    )
    assert [
        column
        for column in computed
        if column not in DefenderPointsPredictor.FEATURES
    ] == []


def test_own_team_and_opposition_form_are_not_swapped(
    predictor, connection, seed_model_frame
) -> None:
    """The gw2 row carries Man Utd's own record, not Arsenal's.

    A cartesian fan-out is caught by the row count; a transposed
    own/opposition join (``own.team = opp_id.team`` instead of
    ``own.team = pw.team``, and vice versa) is caught because Man Utd's
    and Arsenal's gw1 records are deliberately different -- if the two
    sides were swapped, this would fail rather than coincidentally pass.
    """
    seed_model_frame(connection, POSITION)
    row = predictor.build_training_data().filter(pl.col("gw") == 2)

    assert row.height == 1

    # Man Utd's own gw1 record (vs Spurs): 0.3 xG conceded, 0 goals
    # conceded, a clean sheet kept.
    assert row["xg_against_rolling_5"].item() == pytest.approx(0.3)
    assert row["goals_against_rolling_5"].item() == pytest.approx(0.0)
    assert row["clean_sheet_rolling_5"].item() == pytest.approx(1.0)

    # Arsenal's own gw1 record (vs Chelsea): 1.0 xG for, 1 goal for.
    assert row["xg_for_rolling_5"].item() == pytest.approx(1.0)
    assert row["goals_for_rolling_5"].item() == pytest.approx(1.0)


def test_forward_frame_uses_the_inclusive_player_form_view(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """Player form includes the most recent appearance, not just before it.

    Element 1 recorded 0.1 xG in gw1 and 0.9 in gw2 over 90 minutes
    each. The exclusive view -- which the training frame reads -- carries
    0.1 on the gw2 row, because its window stops one match short. The
    inclusive view carries 0.5, the mean over both. As-of joining the
    exclusive view here would make the gw3 forecast a match stale, so
    this test fails against ``player_match_form`` and passes only
    against ``player_match_form_inclusive``.
    """
    seed_forward_history(connection, POSITION)
    register_feature_views(connection)

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=POSITION)])
    )

    assert frame["xg_per90_rolling_5"].item() == pytest.approx(0.5)


def test_forward_frame_uses_the_inclusive_team_form_views(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """Both teams' form includes their most recent match.

    Man Utd's own record over gw1 and gw2 is 0.6 xG against, 1.0 goals
    against and a 0.5 clean-sheet rate; the exclusive view would report
    gw1 alone, 1.0 / 2.0 / 0.0. Arsenal's attacking record is 2.0 xG for
    and 2.0 goals for inclusive, 1.0 / 1.0 exclusive. Every one of the
    five numbers differs between the two views, so this fails if either
    as-of join reaches for the exclusive view -- and the own/opposition
    values are distinct from each other, so a transposed join fails too.
    """
    seed_forward_history(connection, POSITION)
    register_feature_views(connection)

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=POSITION)])
    )

    assert frame["xg_against_rolling_5"].item() == pytest.approx(0.6)
    assert frame["goals_against_rolling_5"].item() == pytest.approx(1.0)
    assert frame["clean_sheet_rolling_5"].item() == pytest.approx(0.5)
    assert frame["xg_for_rolling_5"].item() == pytest.approx(2.0)
    assert frame["goals_for_rolling_5"].item() == pytest.approx(2.0)
