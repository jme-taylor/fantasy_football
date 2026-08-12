"""What makes the goalkeeper model a goalkeeper model.

Everything the goalkeeper shares with the other positions is exercised
once per position in ``test_points.py``. What is left here is what is
particular to keepers: a feature set built from stats no other position
reads, and a training window narrower than the data, because those stats
do not go back as far as the rest.
"""

from datetime import timedelta

import polars as pl
import pytest

from fantasy_football.features.match_form import (
    GK_FPL_PER90_STATS,
    GK_PER90_STATS,
    goalkeeper_covered_seasons,
    per90_column_name,
)
from fantasy_football.modelling.defender import (
    DEFENDER_SPEC,
    DefenderPointsPredictor,
)
from fantasy_football.modelling.folds import ExpandingGameweekFoldStrategy
from fantasy_football.modelling.forwards import FORWARD_SPEC
from fantasy_football.modelling.goalkeeper import (
    GOALKEEPER_SPEC,
    POSITION,
    GoalkeeperPointsPredictor,
)
from fantasy_football.modelling.midfielder import MIDFIELDER_SPEC
from fantasy_football.storage.tables import PLAYER_MATCH, PLAYER_SEASON
from tests.unit.modelling.conftest import (
    ARSENAL,
    GW1_KICKOFF,
    SEASON,
    TEAM_IDS,
    append_rows,
)

# A season the keeper stats predate. FCI opens at 2024-25, so nothing
# earlier can carry goals prevented or xGOT faced.
UNCOVERED_SEASON = "2023-24"


@pytest.fixture
def predictor(connection) -> GoalkeeperPointsPredictor:
    """Return a ``GoalkeeperPointsPredictor`` bound to the test connection."""
    return GoalkeeperPointsPredictor(
        experiment_name="test-goalkeeper",
        params={},
        model_spec=GOALKEEPER_SPEC,
        connection=connection,
        fold_strategy=ExpandingGameweekFoldStrategy(),
    )


def test_spec_position_matches_the_class() -> None:
    """The spec's position is what scopes the partition rewrite.

    A spec pointing at a different position from the class that stamps
    the rows would delete another position's partition and write its own
    rows into it.
    """
    assert GOALKEEPER_SPEC.position == POSITION
    assert GoalkeeperPointsPredictor.POSITION == POSITION


def test_position_uses_the_normalised_label() -> None:
    """``GKP`` is collapsed on ingest, so the model must ask for ``GK``.

    ``player_season.position`` never holds the legacy label, so a model
    declaring ``GKP`` would join to nothing and train on an empty frame.
    """
    assert POSITION == "GK"


def test_registered_model_is_its_own() -> None:
    """Every position registers under a distinct name.

    They share ``production`` as an alias, so a shared registered name
    would have one position's promotion silently serve another.
    """
    assert (
        GOALKEEPER_SPEC.registered_model_name == "goalkeeper_points_regressor"
    )
    assert GOALKEEPER_SPEC.registered_model_name not in {
        DEFENDER_SPEC.registered_model_name,
        FORWARD_SPEC.registered_model_name,
        MIDFIELDER_SPEC.registered_model_name,
    }


def test_every_keeper_stat_reaches_the_feature_list() -> None:
    """Each keeper stat is windowed and then actually read.

    The keeper stats exist only for this model. One computed by the view
    and left out of the feature list would be pure cost.
    """
    for stat in (*GK_PER90_STATS, *GK_FPL_PER90_STATS):
        assert per90_column_name(stat, 5) in GoalkeeperPointsPredictor.FEATURES


def test_outfield_attacking_and_defensive_rates_are_excluded() -> None:
    """A keeper records essentially none of these, so none are read.

    On a training set a fifth the size of the midfielder's, features that
    are structurally zero are overfitting surface, not harmless noise.
    """
    for column in (
        "xg_per90_rolling_5",
        "xa_per90_rolling_5",
        "tackles_per90_rolling_5",
        "interceptions_per90_rolling_5",
        "clearances_per90_rolling_5",
        "blocks_per90_rolling_5",
    ):
        assert column not in GoalkeeperPointsPredictor.FEATURES


def test_team_form_is_the_defender_split() -> None:
    """The keeper reads what his club concedes and what the other creates.

    A keeper's whole points profile is defence-side, so unlike the
    midfielder he takes one half of the team picture -- the same half as
    the defender.
    """
    assert GoalkeeperPointsPredictor.OWN_TEAM_COLUMNS == [
        "xg_against_rolling_5",
        "goals_against_rolling_5",
        "clean_sheet_rolling_5",
    ]
    assert GoalkeeperPointsPredictor.OPPOSITION_COLUMNS == [
        "xg_for_rolling_5",
        "goals_for_rolling_5",
    ]


def test_opposition_columns_are_prefixed_despite_being_disjoint(
    predictor,
) -> None:
    """The prefix is set for legibility, not because a clash forces it.

    The defender takes the same two lists and needs no prefix; this model
    has no registered version to rename, so it can afford one.
    """
    assert predictor.opposition_feature_names == [
        "opposition_xg_for_rolling_5",
        "opposition_goals_for_rolling_5",
    ]
    assert not set(predictor.opposition_feature_names) & set(
        GoalkeeperPointsPredictor.OWN_TEAM_COLUMNS
    )
    assert DefenderPointsPredictor.OPPOSITION_PREFIX == ""


def test_features_include_the_minutes_bucket_probabilities() -> None:
    """The keeper model reads the full minutes distribution.

    The clean sheet is worth four points to a keeper and needs sixty
    minutes on the pitch, so how the minutes fall matters rather than the
    expectation alone.
    """
    for column in ("p_zero", "p_partial", "p_sixty_plus"):
        assert column in GoalkeeperPointsPredictor.MINUTES_COLUMNS
        assert column in GoalkeeperPointsPredictor.FEATURES


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


def test_training_window_is_the_keeper_stat_coverage() -> None:
    """The window is derived, so a narrower stat narrows it on its own.

    Hardcoding it would go stale the moment a keeper stat with different
    coverage is added, and the failure would be silent: the model would
    fit on median-filled values rather than error.
    """
    assert (
        GoalkeeperPointsPredictor.TRAINING_SEASONS
        == goalkeeper_covered_seasons()
    )
    assert "2024-25" in GoalkeeperPointsPredictor.TRAINING_SEASONS
    assert "2023-24" not in GoalkeeperPointsPredictor.TRAINING_SEASONS


def test_only_the_goalkeeper_restricts_its_training_seasons() -> None:
    """The three registered models must keep the frame they were fit on."""
    assert DefenderPointsPredictor.TRAINING_SEASONS is None
    assert GoalkeeperPointsPredictor.TRAINING_SEASONS is not None


def test_training_frame_drops_seasons_before_the_keeper_stats_exist(
    predictor, connection, seed_model_frame
) -> None:
    """A played fixture predating the keeper stats never reaches training.

    Without the restriction it would: the frame takes every played leg,
    and the imputer would median-fill saves, goals prevented and xGOT
    faced for it. The model would fit happily on those rows and
    cross-validation would report nothing wrong.
    """
    seed_model_frame(connection, POSITION)
    append_rows(
        PLAYER_SEASON,
        connection,
        [{"season": UNCOVERED_SEASON, "element": 1, "position": POSITION}],
    )
    append_rows(
        PLAYER_MATCH,
        connection,
        [
            {
                "season": UNCOVERED_SEASON,
                "gw": 38,
                "element": 1,
                "opponent": TEAM_IDS[ARSENAL],
                "is_home": True,
                "minutes": 90,
                "kickoff_time": GW1_KICKOFF - timedelta(days=400),
            }
        ],
    )

    seasons = set(predictor.build_training_data()["season"].to_list())

    assert UNCOVERED_SEASON not in seasons
    assert seasons == {SEASON}


def test_own_team_and_opposition_form_are_not_swapped(
    predictor, connection, seed_model_frame
) -> None:
    """The gw2 row carries each club's record on its own side.

    Element 1 is a Man Utd keeper facing Arsenal in gw2. Man Utd's gw1
    record (vs Spurs) is 0.3 xG against, none conceded, a clean sheet;
    Arsenal's (vs Chelsea) is 1.0 xG for and 1 goal for. The two sides
    differ, so a transposed join fails here rather than passing by
    coincidence.
    """
    seed_model_frame(connection, POSITION)
    row = predictor.build_training_data().filter(pl.col("gw") == 2)

    assert row.height == 1
    assert row["xg_against_rolling_5"].item() == pytest.approx(0.3)
    assert row["goals_against_rolling_5"].item() == pytest.approx(0.0)
    assert row["clean_sheet_rolling_5"].item() == pytest.approx(1.0)
    assert row["opposition_xg_for_rolling_5"].item() == pytest.approx(1.0)
    assert row["opposition_goals_for_rolling_5"].item() == pytest.approx(1.0)
