"""What makes the midfielder model a midfielder model.

Everything the midfielder shares with the other positions is exercised
once per position in ``test_points.py``. What is left here is the column
semantics: a midfielder scores at both ends, so unlike the defender and
the forward -- each of which takes one half of the team picture -- this
model reads every team measure for both clubs. That is only possible
because the opposition copies are prefixed; without the prefix the two
sides would collide on one column name.
"""

import polars as pl
import pytest

from fantasy_football.features.views import register_feature_views
from fantasy_football.modelling.defender import (
    DEFENDER_SPEC,
    DefenderPointsPredictor,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy
from fantasy_football.modelling.forwards import (
    FORWARD_SPEC,
    ForwardPointsPredictor,
)
from fantasy_football.modelling.midfielder import (
    MIDFIELDER_SPEC,
    POSITION,
    MidfielderPointsPredictor,
)

TEAM_MEASURES = [
    "xg_for_rolling_5",
    "goals_for_rolling_5",
    "xg_against_rolling_5",
    "goals_against_rolling_5",
    "clean_sheet_rolling_5",
]


@pytest.fixture
def predictor(connection) -> MidfielderPointsPredictor:
    """Return a ``MidfielderPointsPredictor`` bound to the test connection."""
    return MidfielderPointsPredictor(
        experiment_name="test-midfielder",
        params={},
        model_spec=MIDFIELDER_SPEC,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def test_spec_position_matches_the_class() -> None:
    """The spec's position is what scopes the partition rewrite.

    A spec pointing at a different position from the class that stamps
    the rows would delete another position's partition and write its own
    rows into it.
    """
    assert MIDFIELDER_SPEC.position == POSITION
    assert MidfielderPointsPredictor.POSITION == POSITION


def test_registered_model_is_its_own() -> None:
    """Every position registers under a distinct name.

    They share ``production`` as an alias, so a shared registered name
    would have one position's promotion silently serve another.
    """
    assert (
        MIDFIELDER_SPEC.registered_model_name == "midfielder_points_regressor"
    )
    assert MIDFIELDER_SPEC.registered_model_name not in {
        DEFENDER_SPEC.registered_model_name,
        FORWARD_SPEC.registered_model_name,
    }


def test_both_sides_carry_every_team_measure() -> None:
    """The midfielder takes all five measures for both clubs.

    The defender takes what his club concedes and the forward what his
    club creates; the midfielder scores at both ends, so it reads both
    halves and leaves the trade-off to the model.
    """
    assert MidfielderPointsPredictor.OWN_TEAM_COLUMNS == TEAM_MEASURES
    assert MidfielderPointsPredictor.OPPOSITION_COLUMNS == TEAM_MEASURES


def test_opposition_columns_are_prefixed_in_the_feature_names(
    predictor,
) -> None:
    """The two sides reach the frame under different names.

    Both come from ``team_match_form``, so without the prefix the
    training SELECT would emit two columns called ``xg_for_rolling_5``
    and one side would be lost. The defender and forward models keep an
    empty prefix, so their feature names -- and therefore their
    registered models -- are untouched.
    """
    predictor_names = predictor.opposition_feature_names
    assert predictor_names == [
        f"opposition_{column}" for column in TEAM_MEASURES
    ]
    assert not set(predictor_names) & set(
        MidfielderPointsPredictor.OWN_TEAM_COLUMNS
    )
    assert DefenderPointsPredictor.OPPOSITION_PREFIX == ""
    assert ForwardPointsPredictor.OPPOSITION_PREFIX == ""


def test_features_include_the_minutes_bucket_probabilities() -> None:
    """The midfielder model reads the full minutes distribution.

    A midfielder's clean-sheet point needs sixty minutes on the pitch,
    as a defender's does, so how the minutes are distributed matters
    here rather than the expectation alone.
    """
    for column in ("p_zero", "p_partial", "p_sixty_plus"):
        assert column in MidfielderPointsPredictor.MINUTES_COLUMNS
        assert column in MidfielderPointsPredictor.FEATURES


def test_every_computed_column_reaches_the_feature_list(predictor) -> None:
    """No form column is computed and then silently dropped."""
    computed = (
        predictor.PLAYER_FORM_COLUMNS
        + predictor.OWN_TEAM_COLUMNS
        + predictor.opposition_feature_names
        + predictor.MINUTES_COLUMNS
    )
    assert [
        column for column in computed if column not in predictor.FEATURES
    ] == []


def test_own_team_and_opposition_form_are_not_swapped(
    predictor, connection, seed_model_frame
) -> None:
    """The gw2 row carries both clubs' records, each on its own side.

    Element 1 is a Man Utd midfielder playing Arsenal in gw2. Man Utd's
    gw1 record (vs Spurs) is 2.0 xG for, 2 goals for, 0.3 xG against,
    none conceded and a clean sheet; Arsenal's (vs Chelsea) is 1.0 xG
    for, 1 goal for, 0.5 xG against, 1 conceded and no clean sheet.
    Every pair differs, so a transposed own/opposition join fails here
    rather than coincidentally passing.
    """
    seed_model_frame(connection, POSITION)
    row = predictor.build_training_data().filter(pl.col("gw") == 2)

    assert row.height == 1

    assert row["xg_for_rolling_5"].item() == pytest.approx(2.0)
    assert row["goals_for_rolling_5"].item() == pytest.approx(2.0)
    assert row["xg_against_rolling_5"].item() == pytest.approx(0.3)
    assert row["goals_against_rolling_5"].item() == pytest.approx(0.0)
    assert row["clean_sheet_rolling_5"].item() == pytest.approx(1.0)

    assert row["opposition_xg_for_rolling_5"].item() == pytest.approx(1.0)
    assert row["opposition_goals_for_rolling_5"].item() == pytest.approx(1.0)
    assert row["opposition_xg_against_rolling_5"].item() == pytest.approx(0.5)
    assert row["opposition_goals_against_rolling_5"].item() == pytest.approx(
        1.0
    )
    assert row["opposition_clean_sheet_rolling_5"].item() == pytest.approx(0.0)


def test_forward_frame_uses_the_inclusive_player_form_view(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """Player form includes the most recent appearance, not just before it.

    Element 1 recorded 0.1 xG in gw1 and 0.9 in gw2 over 90 minutes
    each. The exclusive view -- which the training frame reads -- carries
    0.1 on the gw2 row, because its window stops one match short. The
    inclusive view carries 0.5, the mean over both. As-of joining the
    exclusive view here would make the gw3 forecast a match stale.
    """
    seed_forward_history(connection, POSITION)
    register_feature_views(connection)

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=POSITION)])
    )

    assert frame["xg_per90_rolling_5"].item() == pytest.approx(0.5)


def test_forward_frame_carries_both_clubs_on_the_right_side(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """Man Utd's record lands unprefixed and Arsenal's prefixed.

    Inclusive over gw1 and gw2, Man Utd make 0.5 xG and 0.5 goals and
    concede 0.6 xG and 1.0 goals at a 0.5 clean-sheet rate; Arsenal make
    2.0 and 2.0 and concede 0.45 and 0.5. The exclusive views would give
    gw1 alone -- 0.1 / 0.0 / 1.0 / 2.0 for Man Utd -- so this fails on a
    stale view and on a transposed join alike.
    """
    seed_forward_history(connection, POSITION)
    register_feature_views(connection)

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=POSITION)])
    )

    assert frame["xg_for_rolling_5"].item() == pytest.approx(0.5)
    assert frame["goals_for_rolling_5"].item() == pytest.approx(0.5)
    assert frame["xg_against_rolling_5"].item() == pytest.approx(0.6)
    assert frame["goals_against_rolling_5"].item() == pytest.approx(1.0)
    assert frame["clean_sheet_rolling_5"].item() == pytest.approx(0.5)

    assert frame["opposition_xg_for_rolling_5"].item() == pytest.approx(2.0)
    assert frame["opposition_goals_for_rolling_5"].item() == pytest.approx(2.0)
    assert frame["opposition_xg_against_rolling_5"].item() == pytest.approx(
        0.45
    )
    assert frame["opposition_goals_against_rolling_5"].item() == pytest.approx(
        0.5
    )
    assert frame["opposition_clean_sheet_rolling_5"].item() == pytest.approx(
        0.5
    )
