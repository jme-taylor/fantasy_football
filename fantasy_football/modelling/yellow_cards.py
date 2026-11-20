import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, override

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss

from fantasy_football.features.match_form import NO_FORM_COLUMN
from fantasy_football.modelling.components import (
    YELLOW_CARD_POINTS,
    Component,
    YellowCardsComponent,
)
from fantasy_football.modelling.defcon import (
    MINUTES_FLOOR as DEFCON_MINUTES_FLOOR,
)
from fantasy_football.modelling.distributions import PoissonCounts
from fantasy_football.modelling.metrics import (
    count_poisson_deviance,
    top_decile_ratio,
)
from fantasy_football.modelling.points import (
    PositionPointsPredictor,
    position_dummy_names,
)
from fantasy_football.modelling.predictor import ModelSpec
from fantasy_football.storage.coverage import FCI_SEASONS, FPL_STAT_SEASONS
from fantasy_football.storage.tables import (
    POINTS_COMPONENT,
    TEST_POINTS_PREDICTION,
)

logger = logging.getLogger(__name__)

#: The model's primary position, which keys its registered name and
#: its evaluation. It scores and writes every position in
#: :data:`TRAINING_POSITIONS` -- see ``SERVING_POSITIONS``.
POSITION = "DEF"

#: The outfield positions the one artefact is fitted on, and
#: writes component rows for.
TRAINING_POSITIONS: tuple[str, ...] = ("DEF", "MID", "FWD")

REGISTERED_MODEL = "yellow_cards_rate_regressor"
PRODUCTION_ALIAS = "production"
EXPERIMENT_NAME = "yellow-cards-rate-model"

#: Every season a card was published in -- ten of them, against the two
#: the FCI-fed heads get. See the module docstring.
TRAINING_SEASONS = FPL_STAT_SEASONS

#: Where the holdout is allowed to land. Restricted to the seasons whose
#: feature set matches what serving actually sees.
TEST_SEASONS = FCI_SEASONS

# No training floor, unlike the goals and assists heads. Their reasoning
# inverts here: a booking inside a cameo is a real and informative
# observation, not arithmetic noise, and ``WEIGHT_COLUMN`` already cuts
# its leverage to almost nothing. The scoring floor still matches the
# sibling components' -- below it a short appearance would lose its whole
# composed prediction rather than just its cards term.
SCORING_MINUTES_FLOOR = DEFCON_MINUTES_FLOOR


def yellow_card_points_sql(match: str = "m") -> str:
    """Return the points a leg's yellow cards cost.

    Negative, because that is what the points are: a booking costs
    this expression and so adds the cost back to what it must explain.

    Aliased rather than fixed so a caller can point it at whichever
    same expression off its own joins. One definition, two aliasings --
    a second spelling of this is how the components quietly stop summing
    to the total.

    Parameters
    ----------
    match : str, optional
        Alias of the ``player_match`` relation.

    Returns
    -------
    str
        A scalar expression, unaliased.
    """
    return f"{YELLOW_CARD_POINTS} * coalesce({match}.yellow_cards, 0)"


@dataclass(frozen=True, slots=True)
class YellowCardsMetrics:
    """Scores for the yellow-cards head over one fold.

    Scored at match scale rather than on the per-90 rate, for the reason
    the goals and assists heads are: almost every row is a zero, and a
    rate error is minimised by predicting nothing at all.

    ``player_rate_skill`` is the one that matters. Beating the league
    base rate on cards is nearly free, because per-player memory is most
    of the available signal; beating each player's own trailing rate is
    what says the model learned something. It is a diagnostic, not a
    gate -- nothing here blocks promotion.
    """

    poisson_deviance: float
    brier: float
    base_rate_brier: float
    skill_score: float
    player_rate_skill: float
    booking_rate: float
    rate_mae: float
    top_decile_ratio: float

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, one level deep."""
        return asdict(self)


class YellowCardsRatePredictor(PositionPointsPredictor):
    """Predicts yellow cards per 90 for outfielders, minutes-blind."""

    POSITION = POSITION
    TRAINING_POSITIONS = TRAINING_POSITIONS
    SERVING_POSITIONS = TRAINING_POSITIONS
    TARGET = "yellow_cards_per_90"
    COMPONENT = Component.YELLOW_CARDS
    COMPONENT_IMPL = YellowCardsComponent()
    TRAINING_SEASONS = TRAINING_SEASONS
    WEIGHT_COLUMN = "minutes"
    EXTRA_COLUMNS = (
        "m.yellow_cards AS yellow_cards",
        "m.minutes AS minutes",
    )
    # A null flag means the form join missed entirely, which is the same
    # thing the flag exists to report. Left to the imputer it would come
    # back as the population median instead. Unlike the sibling heads
    # this head keeps those rows and reads the flag as a feature, so the
    # fill is what makes the long training window work at all.
    FEATURE_FILLS: ClassVar[dict[str, float]] = {NO_FORM_COLUMN: 1.0}

    # Narrow on purpose. ``fouls_committed`` is the mechanism a booking
    # comes from and ``tackles`` the exposure that produces fouls; the
    # season-to-date count is a genuinely different signal from the
    # rolling rate, since a player on four yellows is refereed
    # differently from one on none. The red rate is here despite reds not
    # being the target: a player who gets sent off also gets booked.
    #
    # FCI's fouls suffered, duels lost and dribbled past are all
    # plausibly card-adjacent on the theory that a beaten defender fouls
    # to recover. They are left for a second pass rather than spent now.
    FEATURES = [
        "is_home",
        *position_dummy_names(TRAINING_POSITIONS),
        NO_FORM_COLUMN,
        "yellow_cards_per90_rolling_5",
        "yellow_cards_season_to_date",
        "red_cards_per90_rolling_5",
        "fouls_committed_per90_rolling_5",
        "tackles_per90_rolling_5",
        "xg_against_rolling_5",
    ]

    PLAYER_FORM_COLUMNS = [
        NO_FORM_COLUMN,
        "yellow_cards_per90_rolling_5",
        "yellow_cards_season_to_date",
        "red_cards_per90_rolling_5",
        "fouls_committed_per90_rolling_5",
        "tackles_per90_rolling_5",
    ]
    OWN_TEAM_COLUMNS: ClassVar[list[str]] = []
    # How much defending the side is made to do. A team pinned back
    # commits more fouls, and its defenders take more of the bookings.
    OPPOSITION_COLUMNS = ["xg_against_rolling_5"]
    MINUTES_COLUMNS: ClassVar[list[str]] = []

    @property
    @override
    def target_sql(self) -> str:
        """Return yellow cards per 90, as FPL published them."""
        return (
            "m.yellow_cards * 90.0 "
            "/ nullif(m.minutes, 0) AS yellow_cards_per_90"
        )

    @property
    @override
    def row_filter(self) -> str:
        """Match the other DEF components' floor, and nothing tighter.

        Every restriction here removes a leg from the composed
        prediction, not just from the fit: a defender missing only his
        yellow-cards component is incomplete, and ``compose`` drops him
        entirely.
        """
        return f"\n  AND m.minutes >= {SCORING_MINUTES_FLOOR}"

    @property
    @override
    def training_row_filter(self) -> str:
        """Drop only legs with no card count behind them.

        No minutes floor and no no-form filter, which is where this head
        parts company with goals and assists. Both of those exclusions
        would be right for a head whose features are all FCI columns and
        whose target explodes on short appearances; here the first would
        throw away eight of ten training seasons and the second would
        throw away the cameo bookings that are the most informative rows
        in the set. Minutes weighting handles the leverage the floor
        exists for.

        A null count is a leg no source filed cards for, which has no
        target at all -- distinct from a zero, which is a player who
        played and was not booked.
        """
        return "\n  AND m.yellow_cards IS NOT NULL"

    @override
    def fold_metrics(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> YellowCardsMetrics:
        """Score the fold at match scale, not on the per-90 rate.

        Actual minutes build the expected count rather than the minutes
        model's forecast, which isolates this head's error from that
        model's.
        """
        minutes = test_df["minutes"].cast(pl.Float64).to_numpy()
        rate = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
        expected = rate * minutes / 90.0
        cards = test_df["yellow_cards"].cast(pl.Float64).to_numpy()
        booked = (cards > 0).astype(int)
        probability = PoissonCounts().p_at_least(expected, 1)
        base_rate = float(np.mean(booked))
        baseline = np.full_like(probability, base_rate)
        brier = float(brier_score_loss(booked, probability, pos_label=1))
        base_brier = float(brier_score_loss(booked, baseline, pos_label=1))
        return YellowCardsMetrics(
            poisson_deviance=count_poisson_deviance(cards, expected),
            brier=brier,
            base_rate_brier=base_brier,
            skill_score=1.0 - brier / base_brier if base_brier else 0.0,
            player_rate_skill=self._player_rate_skill(
                test_df, booked, probability, brier
            ),
            booking_rate=base_rate,
            rate_mae=float(
                np.average(
                    np.abs(rate - test_df[self.TARGET].to_numpy()),
                    weights=minutes,
                )
            ),
            top_decile_ratio=top_decile_ratio(rate, cards, expected),
        )

    def _player_rate_skill(
        self,
        test_df: pl.DataFrame,
        booked: np.ndarray,
        probability: np.ndarray,
        brier: float,
    ) -> float:
        """Return the Brier skill against each player's own trailing rate.

        The trailing rate is the feature the model already reads, turned
        into a booking probability the same way its own prediction is.
        A player with no history behind him falls back to the fold's
        base rate, which is what an imputer would have given him anyway.
        """
        minutes = test_df["minutes"].cast(pl.Float64).to_numpy()
        trailing = (
            test_df["yellow_cards_per90_rolling_5"]
            .cast(pl.Float64)
            .fill_null(float(np.mean(booked)))
            .to_numpy()
        )
        naive = PoissonCounts().p_at_least(
            np.clip(trailing, 0.0, None) * minutes / 90.0, 1
        )
        naive_brier = float(brier_score_loss(booked, naive, pos_label=1))
        return 1.0 - brier / naive_brier if naive_brier else 0.0


YELLOW_CARDS_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_POINTS_PREDICTION,
    position=POSITION,
    component=Component.YELLOW_CARDS,
)
