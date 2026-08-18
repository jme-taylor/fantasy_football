"""The yellow-card rate model, pooled across the outfield positions.

A yellow costs one point whatever the shirt, so like the assists head
this needs no per-position conversion. It pools for the same reason too:
a booking is rare enough for one defender that his own history barely
estimates a rate, while the process behind it -- commit a foul, dissent,
waste time -- is the same one a midfielder's comes from. ``POSITION``
says only which position's rows this instance scores and writes.

Goalkeepers are excluded. They are booked, but for different reasons and
at a fraction of the rate, and their defensive counters describe
something structurally different.

Reds are not modelled and stay in the residual. They pay minus three
against a yellow's minus one and arrive roughly forty times less often,
which is too thin to fit a player rate to. This head is named for
yellows so that a red component, if it ever earns one, has somewhere to
go that is not a nullable field in here.

Cards come from ``player_match`` rather than from a provider table.
Vaastav's per-fixture files stop at 2025-26 and FCI has never published
cards at all, so a provider-sourced card column is null for the live
season -- which is what the spine now carries them for. That is also
why this head needs no extra join: the count is on the match row.

Like its sibling heads this reads no minutes features. It predicts a
rate and minutes turn that rate into an expected count at composition,
so the minutes forecast is applied exactly once across every component.

The residual model deducts the same expression through
:func:`yellow_card_points_sql`, and the two must not diverge: if the
target and the deduction disagree about what a booking costs, the
components stop summing to the total and nothing fails loudly.

Two deliberate departures from the sibling heads:

Training reaches back to 2016-17 rather than stopping at the FCI
seasons. Cards are published six seasons further back than any FCI
column and the event is rare, so that history is the most valuable thing
available. Eight of those ten seasons carry no FCI column at all, and
``has_no_form`` -- anchored on xG, which FCI files for every appearance
-- already marks them, so the regime is flagged rather than hidden. The
holdout is restricted to the FCI seasons regardless: a metric averaged
over a regime containing 2017 says nothing about how this does on
Saturday.

Consequently, and unlike goals and assists, no-form rows are *kept* in
the fit. Excluding them is right for a head whose features are all FCI
columns; it would throw away eight of ten seasons here.

Known gap: a red card costs the player his next match, which is worth
far more than the minus three the card itself costs, and the minutes
model knows nothing about it. That belongs in the minutes model rather
than here.

Promotion order: promote the *residual* alias before this one. The two
carry separately promoted aliases and between the two promotions the
composed total is wrong either way, but under-charging bookings briefly
is the smaller error.
"""

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

#: The position this instance scores and writes. The model behind it
#: knows about all three in :data:`TRAINING_POSITIONS`.
POSITION = "DEF"

#: The outfield positions the one artefact is fitted on.
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

    Negative, because that is what the points are: the residual deducts
    this expression and so adds the cost back to what it must explain.

    Aliased rather than fixed so the residual model can deduct the very
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
