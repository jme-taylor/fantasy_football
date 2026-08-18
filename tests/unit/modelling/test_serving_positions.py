"""Which positions a pooled model writes rows for.

This is the one place a mistake is silent rather than loud. A row
stamped with the wrong position still names a legitimate component, so
``compose``'s stale-component guard passes it through and the player is
simply priced as something he is not.
"""

from typing import ClassVar

import polars as pl
import pytest

from fantasy_football.modelling.assists import AssistsRatePredictor
from fantasy_football.modelling.defcon import (
    CbirtRatePredictor,
    DefconRatePredictor,
)
from fantasy_football.modelling.goals import GoalsRatePredictor
from fantasy_football.modelling.points import PositionPointsPredictor
from fantasy_football.modelling.yellow_cards import YellowCardsRatePredictor


class _Stub(PositionPointsPredictor):
    """A predictor built only to read its position properties."""

    POSITION = "DEF"
    FEATURES: ClassVar[list[str]] = []
    PLAYER_FORM_COLUMNS: ClassVar[list[str]] = []
    OWN_TEAM_COLUMNS: ClassVar[list[str]] = []
    OPPOSITION_COLUMNS: ClassVar[list[str]] = []
    MINUTES_COLUMNS: ClassVar[list[str]] = []

    def __init__(self) -> None:
        """Skip the base constructor; nothing here touches a database."""


def test_a_single_position_model_serves_only_itself() -> None:
    """The default is unchanged: one position, stamped as a literal."""
    stub = _Stub()

    assert stub.serving_positions == ("DEF",)
    assert stub._served_position().meta.eq(pl.lit("DEF"))


def test_serving_a_position_never_trained_on_is_refused() -> None:
    """The dummy telling positions apart would not exist for it."""

    class Unseen(_Stub):
        TRAINING_POSITIONS = ("DEF", "MID")
        SERVING_POSITIONS = ("DEF", "MID", "FWD")

    with pytest.raises(ValueError, match="FWD"):
        _ = Unseen().serving_positions


def test_the_stamp_reads_the_row_and_not_the_primary_position() -> None:
    """Each row is stamped with the position its dummy reports."""

    class Pooled(_Stub):
        TRAINING_POSITIONS = ("DEF", "MID", "FWD")
        SERVING_POSITIONS = ("DEF", "MID", "FWD")

    rows = pl.DataFrame(
        {
            "is_defender": [1.0, 0.0, 0.0],
            "is_midfielder": [0.0, 1.0, 0.0],
            "is_forward": [0.0, 0.0, 1.0],
        }
    )

    stamped = rows.with_columns(position=Pooled()._served_position())

    assert stamped["position"].to_list() == ["DEF", "MID", "FWD"]


def test_the_pooled_heads_serve_every_position_they_train_on() -> None:
    """A head fitted on three positions that serves one wastes two."""
    for cls in (
        GoalsRatePredictor,
        AssistsRatePredictor,
        YellowCardsRatePredictor,
    ):
        assert cls.SERVING_POSITIONS == ("DEF", "MID", "FWD")


def test_the_two_defcon_heads_split_the_positions_between_them() -> None:
    """Different counts and thresholds, so neither may serve the other."""
    assert DefconRatePredictor.SERVING_POSITIONS == ()
    assert DefconRatePredictor.POSITION == "DEF"
    assert CbirtRatePredictor.SERVING_POSITIONS == ("MID", "FWD")
