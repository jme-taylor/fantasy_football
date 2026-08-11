"""What makes the forwards model a forwards model.

Everything the forwards model shares with the other positions is
exercised once per position in ``test_points.py``. What is left here is
the column semantics, which are the mirror image of the defender's: a
forward is scored on what his own club creates and on what the
opposition concedes.
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
    POSITION,
    ForwardPointsPredictor,
)


@pytest.fixture
def predictor(connection) -> ForwardPointsPredictor:
    """Return a ``ForwardPointsPredictor`` bound to the test connection."""
    return ForwardPointsPredictor(
        experiment_name="test-forwards",
        params={},
        model_spec=FORWARD_SPEC,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def test_spec_position_matches_the_class() -> None:
    """The spec's position is what scopes the partition rewrite.

    A spec pointing at a different position from the class that stamps
    the rows would delete another position's partition and write its own
    rows into it.
    """
    assert FORWARD_SPEC.position == POSITION
    assert ForwardPointsPredictor.POSITION == POSITION


def test_registered_model_is_not_the_defenders() -> None:
    """The two positions register under distinct names.

    They share ``production`` as an alias, so a shared registered name
    would have one position's promotion silently serve the other.
    """
    assert (
        FORWARD_SPEC.registered_model_name
        != DEFENDER_SPEC.registered_model_name
    )
    assert FORWARD_SPEC.registered_model_name == "forwards_points_regressor"


def test_own_columns_are_attacking_and_opposition_columns_defensive() -> None:
    """A forward's own record is what his club creates.

    This is the exact mirror of the defender model, which is the point:
    the shared base reads the same two lists for both, so getting them
    the wrong way round is the failure mode worth pinning.
    """
    assert ForwardPointsPredictor.OWN_TEAM_COLUMNS == [
        "xg_for_rolling_5",
        "goals_for_rolling_5",
    ]
    assert ForwardPointsPredictor.OPPOSITION_COLUMNS == [
        "xg_against_rolling_5",
        "goals_against_rolling_5",
    ]
    assert (
        ForwardPointsPredictor.OWN_TEAM_COLUMNS
        != DefenderPointsPredictor.OWN_TEAM_COLUMNS
    )


def test_features_omit_the_minutes_bucket_probabilities() -> None:
    """The forwards model deliberately takes expected_minutes alone.

    Unlike a clean sheet, attacking returns scale roughly with time on
    the pitch, so the bucket split adds little over the expectation.
    Pinned because the shared base generates the minutes SELECT from
    MINUTES_COLUMNS -- if this ever grows, it should be a decision.
    """
    assert ForwardPointsPredictor.MINUTES_COLUMNS == ["expected_minutes"]
    for column in ("p_zero", "p_partial", "p_sixty_plus"):
        assert column not in ForwardPointsPredictor.FEATURES


def test_own_team_and_opposition_form_are_not_swapped(
    predictor, connection, seed_model_frame
) -> None:
    """The gw2 row carries Man Utd's own attacking record, not Arsenal's.

    Element 1 is a Man Utd forward playing Arsenal in gw2. Man Utd's
    gw1 attacking record (vs Spurs) is 2.0 xG for and 2 goals; Arsenal's
    (vs Chelsea) is 1.0 and 1. The two are deliberately different, so a
    transposed own/opposition join fails here rather than coincidentally
    passing.
    """
    seed_model_frame(connection, POSITION)
    row = predictor.build_training_data().filter(pl.col("gw") == 2)

    assert row.height == 1
    assert row["xg_for_rolling_5"].item() == pytest.approx(2.0)
    assert row["goals_for_rolling_5"].item() == pytest.approx(2.0)


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


def test_forward_frame_uses_the_inclusive_own_team_form_view(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """The own-club attacking record includes Man Utd's most recent match.

    Man Utd make 0.1 xG and no goals in gw1, then 0.9 and 1 in gw2:
    inclusive 0.5 / 0.5, exclusive 0.1 / 0.0. Both numbers differ
    between the views, so this fails if the as-of join reaches for the
    exclusive one.
    """
    seed_forward_history(connection, POSITION)
    register_feature_views(connection)

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=POSITION)])
    )

    assert frame["xg_for_rolling_5"].item() == pytest.approx(0.5)
    assert frame["goals_for_rolling_5"].item() == pytest.approx(0.5)


def test_every_computed_column_reaches_the_feature_list() -> None:
    """No form column is computed and then silently dropped.

    The opposition's defensive record used to be joined and then thrown
    away, because FEATURES listed neither column -- so the model could
    not tell a trip to the best defence in the league from a home tie
    against the worst.
    """
    computed = (
        ForwardPointsPredictor.PLAYER_FORM_COLUMNS
        + ForwardPointsPredictor.OWN_TEAM_COLUMNS
        + ForwardPointsPredictor.OPPOSITION_COLUMNS
        + ForwardPointsPredictor.MINUTES_COLUMNS
    )
    assert [
        column
        for column in computed
        if column not in ForwardPointsPredictor.FEATURES
    ] == []


def test_forward_frame_uses_the_inclusive_opposition_form_view(
    predictor, connection, seed_forward_history, forward_fixture, forward_frame
) -> None:
    """Arsenal's defensive record reaches the forward's feature row.

    Arsenal concede 0.5 xG and 1 goal in gw1, then 0.4 and none in gw2:
    inclusive 0.45 xG against and 0.5 goals against, exclusive 0.5 and
    1.0. Both numbers differ between the views, and both differ from
    Man Utd's own record, so this fails on a stale view and on a
    transposed own/opposition join alike.
    """
    seed_forward_history(connection, POSITION)
    register_feature_views(connection)

    frame = predictor.build_forward_data(
        forward_frame([forward_fixture(position=POSITION)])
    )

    assert frame["xg_against_rolling_5"].item() == pytest.approx(0.45)
    assert frame["goals_against_rolling_5"].item() == pytest.approx(0.5)
