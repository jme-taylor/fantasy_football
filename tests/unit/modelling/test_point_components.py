"""The scoring components a decomposed DEF prediction is built from.

Every component is tested against the same contract: given keyed rows
carrying its model's output and a frame of minutes predictions, it
returns component rows whose points can be hand-computed.
"""

import json
import math

import numpy as np
import polars as pl
import pytest
from scipy.stats import poisson

from fantasy_football.modelling.assists import AssistsRatePredictor
from fantasy_football.modelling.components import (
    ASSIST_POINTS,
    MINUTES_OUTPUTS,
    PREDICTED_VALUE,
    AppearanceComponent,
    Component,
    ConcedingComponent,
    DefconComponent,
    PointsComponent,
    RateComponent,
    SavesComponent,
    expected_floor,
)
from fantasy_football.modelling.defcon import (
    CbirtRatePredictor,
    DefconRatePredictor,
)
from fantasy_football.modelling.distributions import PoissonCounts
from fantasy_football.modelling.goals import GoalsRatePredictor
from fantasy_football.modelling.yellow_cards import YellowCardsRatePredictor
from fantasy_football.storage.tables import BACKFILL_KIND

SEASON = "2025-26"
DEFCON_THRESHOLD = 10


def scored_rows(*values: float, position: str = "DEF") -> pl.DataFrame:
    """Build keyed rows carrying one model output each."""
    return pl.DataFrame(
        {
            "season": [SEASON] * len(values),
            "gw": [1] * len(values),
            "element": list(range(1, len(values) + 1)),
            "opponent": [2] * len(values),
            "position": [position] * len(values),
            PREDICTED_VALUE: list(values),
        }
    )


def minutes_rows(
    *,
    n: int = 1,
    expected_minutes: float = 90.0,
    p_partial: float = 0.1,
    p_sixty_plus: float = 0.9,
) -> pl.DataFrame:
    """Build minutes-model predictions for the same keys."""
    return pl.DataFrame(
        {
            "season": [SEASON] * n,
            "gw": [1] * n,
            "element": list(range(1, n + 1)),
            "opponent": [2] * n,
            "expected_minutes": [expected_minutes] * n,
            "p_zero": [1.0 - p_partial - p_sixty_plus] * n,
            "p_partial": [p_partial] * n,
            "p_sixty_plus": [p_sixty_plus] * n,
        }
    )


def defcon() -> DefconComponent:
    """Return a defcon component over a fitted Poisson."""
    return DefconComponent(
        threshold=DEFCON_THRESHOLD, distribution=PoissonCounts()
    )


# --- The shared contract ---------------------------------------------


@pytest.mark.parametrize(
    "component",
    [
        AppearanceComponent(),
        defcon(),
        ConcedingComponent(),
        RateComponent(Component.GOALS),
    ],
    ids=lambda c: str(c.component),
)
def test_every_component_satisfies_the_protocol(component) -> None:
    """Each implementation is a PointsComponent."""
    assert isinstance(component, PointsComponent)


@pytest.mark.parametrize(
    "component",
    [
        AppearanceComponent(),
        defcon(),
        ConcedingComponent(),
        RateComponent(Component.GOALS),
    ],
    ids=lambda c: str(c.component),
)
def test_every_component_returns_keyed_component_rows(component) -> None:
    """The contract is keyed rows, not a bare column of numbers."""
    rows = component.points(scored_rows(6.0), minutes_rows(), BACKFILL_KIND)

    assert {
        "season",
        "gw",
        "element",
        "opponent",
        "position",
        "prediction_kind",
        "component",
        "points",
        "diagnostics",
    } <= set(rows.columns)
    assert rows["component"].unique().to_list() == [str(component.component)]
    assert rows["prediction_kind"].unique().to_list() == [BACKFILL_KIND]


@pytest.mark.parametrize(
    "component",
    [
        AppearanceComponent(),
        defcon(),
        ConcedingComponent(),
        RateComponent(Component.GOALS),
        SavesComponent(),
    ],
    ids=lambda c: str(c.component),
)
def test_no_component_reads_a_minutes_feature(component) -> None:
    """Minutes reach a component as an argument, never as a feature.

    The load-bearing rule of the decomposition: minutes are applied once,
    at composition. A component that took a minutes column as a model
    feature would double-count it the moment a second component was
    added, and the arithmetic would stop being a decomposition without
    anything failing. There is no exception left: every position is
    decomposed, so no component reads its own minutes.
    """
    assert not set(component.model_features) & set(MINUTES_OUTPUTS)


# --- Appearance ------------------------------------------------------


def test_appearance_pays_one_for_a_cameo_and_two_for_an_hour() -> None:
    """Appearance points are read straight off the minutes model."""
    rows = AppearanceComponent().points(
        scored_rows(0.0),
        minutes_rows(p_partial=0.3, p_sixty_plus=0.6),
        BACKFILL_KIND,
    )

    assert rows["points"].to_list() == pytest.approx([0.3 * 1 + 0.6 * 2])


def test_appearance_ignores_whatever_the_row_carries() -> None:
    """Appearance needs no model, so the scored value is irrelevant."""
    minutes = minutes_rows(p_partial=0.2, p_sixty_plus=0.5)
    high = AppearanceComponent().points(
        scored_rows(99.0), minutes, BACKFILL_KIND
    )
    low = AppearanceComponent().points(
        scored_rows(0.0), minutes, BACKFILL_KIND
    )

    assert high["points"].to_list() == low["points"].to_list()


# --- Defcon ----------------------------------------------------------


def test_defcon_walks_rate_through_lambda_to_points() -> None:
    """Rate, minutes, threshold and payout compose as specified."""
    rate = 12.0
    rows = defcon().points(
        scored_rows(rate), minutes_rows(expected_minutes=45.0), BACKFILL_KIND
    )

    expected_lambda = rate * 45.0 / 90.0
    expected_points = 2.0 * PoissonCounts().p_at_least(
        [expected_lambda], DEFCON_THRESHOLD
    )
    assert rows["points"].to_list() == pytest.approx(list(expected_points))


def test_defcon_cannot_pay_more_than_two_points() -> None:
    """The threshold pays two points, however high the rate."""
    rows = defcon().points(scored_rows(60.0), minutes_rows(), BACKFILL_KIND)

    assert rows["points"].item() <= 2.0


def test_defcon_pays_nothing_to_a_defender_who_will_not_play() -> None:
    """Zero expected minutes is zero expected defcon points."""
    rows = defcon().points(
        scored_rows(12.0), minutes_rows(expected_minutes=0.0), BACKFILL_KIND
    )

    assert rows["points"].item() == pytest.approx(0.0)


def test_defcon_is_monotonic_in_the_rate() -> None:
    """A busier defender is likelier to clear the threshold."""
    rows = defcon().points(
        scored_rows(2.0, 8.0, 14.0), minutes_rows(n=3), BACKFILL_KIND
    )

    points = rows.sort("element")["points"].to_list()
    assert points == sorted(points)


def test_defcon_records_its_intermediates() -> None:
    """The rate, lambda and hit probability are stored for debugging."""
    rows = defcon().points(
        scored_rows(12.0), minutes_rows(expected_minutes=90.0), BACKFILL_KIND
    )

    diagnostics = json.loads(rows["diagnostics"].item())
    assert diagnostics["rate"] == pytest.approx(12.0)
    assert diagnostics["lambda"] == pytest.approx(12.0)
    assert 0.0 <= diagnostics["p_hit"] <= 1.0
    assert rows["points"].item() == pytest.approx(2.0 * diagnostics["p_hit"])


def test_defcon_survives_a_negative_predicted_rate() -> None:
    """A regressor undershooting below zero yields no points, not a nan."""
    rows = defcon().points(scored_rows(-1.0), minutes_rows(), BACKFILL_KIND)

    assert rows["points"].item() == pytest.approx(0.0)


# --- Rate components (the residual) ----------------------------------


# --- Missing minutes -------------------------------------------------


@pytest.mark.parametrize(
    "component",
    [
        AppearanceComponent(),
        defcon(),
        ConcedingComponent(),
        RateComponent(Component.GOALS),
    ],
    ids=lambda c: str(c.component),
)
def test_a_row_with_no_minutes_forecast_scores_zero(component) -> None:
    """An unmatched minutes row gives zero points, never a null.

    A null would survive the sum as a null and silently blank out the
    whole composed prediction for that fixture leg.
    """
    rows = component.points(
        scored_rows(6.0), minutes_rows().head(0), BACKFILL_KIND
    )

    assert rows["points"].to_list() == pytest.approx([0.0])


# --- The guardrail on the real models --------------------------------


@pytest.mark.parametrize(
    "predictor",
    [
        DefconRatePredictor,
        CbirtRatePredictor,
        GoalsRatePredictor,
        AssistsRatePredictor,
        YellowCardsRatePredictor,
    ],
    ids=lambda cls: cls.__name__,
)
def test_no_decomposed_model_takes_a_minutes_feature(predictor) -> None:
    """The load-bearing rule, asserted against the shipped feature lists.

    Minutes are applied once, at composition. A component model that read
    expected_minutes or a bucket probability would have minutes in its
    prediction *and* have them multiplied in again downstream, and the
    arithmetic would stop being a decomposition without any test failing.
    This is the guardrail nobody would think to write and would most
    regret missing.
    """
    assert not set(predictor.FEATURES) & set(MINUTES_OUTPUTS)
    assert not set(predictor.MINUTES_COLUMNS)


def test_a_flat_price_pays_every_position_the_same() -> None:
    """An assist is three points whatever the shirt.

    The mapping form exists to express variation by position, and there
    is none here -- so the scalar has to price a midfielder's rate
    exactly as it prices a defender's.
    """
    component = RateComponent(
        Component.ASSISTS, points_per_event=ASSIST_POINTS
    )

    paid = [
        component.points(
            scored_rows(0.5, position=position),
            minutes_rows(),
            BACKFILL_KIND,
        )["points"].item()
        for position in ("DEF", "MID", "FWD")
    ]

    # Half an assist per 90 over a full match, at three points each.
    assert paid == [pytest.approx(1.5)] * 3


def test_a_position_with_no_price_in_a_mapping_is_an_error() -> None:
    """A scalar prices everyone; a mapping has to be asked to."""
    component = RateComponent(Component.GOALS, points_per_event={"DEF": 6.0})

    with pytest.raises(ValueError, match="no points value"):
        component.points(
            scored_rows(0.5, position="MID"), minutes_rows(), BACKFILL_KIND
        )


# --- Conceding -------------------------------------------------------


def test_conceding_pays_the_clean_sheet_off_the_full_match_rate() -> None:
    """A midfielder's only conceding leg is the clean sheet.

    Hand-computed against ``exp(-rate)``, which is P(no goals) for a
    Poisson arrived at independently of the distribution the component
    walks the rate through.
    """
    rate = 1.2
    rows = ConcedingComponent().points(
        scored_rows(rate, position="MID"),
        minutes_rows(expected_minutes=90.0, p_sixty_plus=0.8),
        BACKFILL_KIND,
    )

    assert rows["points"].item() == pytest.approx(1.0 * 0.8 * math.exp(-rate))


def test_conceding_docks_the_expected_floor_of_half_the_goals() -> None:
    """The deduction is E[floor(N / 2)], not P(N >= 2).

    Hand-computed by summing ``floor(n / 2)`` against the Poisson mass at
    each count -- a different expression from the survival-function sum
    the component uses to reach the same expectation. The clean-sheet leg
    is switched off with ``p_sixty_plus`` so the deduction stands alone.
    """
    rate = 2.0
    rows = ConcedingComponent().points(
        scored_rows(rate),
        minutes_rows(expected_minutes=45.0, p_partial=0.0, p_sixty_plus=0.0),
        BACKFILL_KIND,
    )

    exposed = rate * 45.0 / 90.0
    expected = sum((n // 2) * poisson.pmf(n, exposed) for n in range(0, 30))
    assert rows["points"].item() == pytest.approx(-expected, abs=1e-9)


def test_conceding_docks_more_than_a_single_point_at_a_high_rate() -> None:
    """A defence expected to ship three loses more than one point.

    The specific mistake this guards: paying ``-1 * P(N >= 2)``, which is
    bounded at one point however bad the defence, and which would price a
    disaster fixture the same as a merely poor one.
    """
    rows = ConcedingComponent().points(
        scored_rows(4.0),
        minutes_rows(p_partial=0.0, p_sixty_plus=0.0),
        BACKFILL_KIND,
    )

    assert rows["points"].item() < -1.0


def test_conceding_rejects_a_position_it_has_no_clean_sheet_price_for() -> (
    None
):
    """An unpriced position must fail rather than score null.

    A null points value survives the sum as a null and blanks the whole
    composed prediction, so this is the one failure mode worth raising on.
    """
    rows = scored_rows(1.0).with_columns(position=pl.lit("MNG"))

    with pytest.raises(ValueError, match="no clean sheet value"):
        ConcedingComponent().points(rows, minutes_rows(), BACKFILL_KIND)


def test_conceding_docks_only_the_positions_fpl_docks() -> None:
    """A midfielder keeps the clean sheet but is never docked."""
    paid = {
        position: ConcedingComponent()
        .points(
            scored_rows(3.0, position=position),
            minutes_rows(p_partial=0.0, p_sixty_plus=0.0),
            BACKFILL_KIND,
        )["points"]
        .item()
        for position in ("GK", "DEF", "MID", "FWD")
    }

    assert paid["GK"] == pytest.approx(paid["DEF"])
    assert paid["GK"] < 0.0
    assert paid["MID"] == pytest.approx(0.0)
    assert paid["FWD"] == pytest.approx(0.0)


def test_conceding_pays_a_forward_nothing_at_all() -> None:
    """No clean sheet, no deduction, whatever the fixture."""
    rows = ConcedingComponent().points(
        scored_rows(0.2, position="FWD"), minutes_rows(), BACKFILL_KIND
    )

    assert rows["points"].item() == pytest.approx(0.0)


def test_conceding_reads_full_minutes_for_the_sheet_and_part_for_the_dock() -> (
    None
):
    """The two legs take different exposures, by design.

    Halving expected minutes must halve the deduction's exposure while
    leaving the clean-sheet probability alone -- the asymmetry that stops
    a rotation risk being credited with a better clean-sheet chance than
    the defender who plays every minute of the same fixture.
    """
    rate = 1.5
    full = ConcedingComponent().points(
        scored_rows(rate, position="MID"),
        minutes_rows(expected_minutes=90.0, p_sixty_plus=0.7),
        BACKFILL_KIND,
    )
    half = ConcedingComponent().points(
        scored_rows(rate, position="MID"),
        minutes_rows(expected_minutes=45.0, p_sixty_plus=0.7),
        BACKFILL_KIND,
    )

    # MID is never docked, so any difference could only come from the
    # clean-sheet leg having been scaled.
    assert full["points"].item() == pytest.approx(half["points"].item())

    docked = [
        ConcedingComponent()
        .points(
            scored_rows(rate),
            minutes_rows(
                expected_minutes=expected, p_partial=0.0, p_sixty_plus=0.0
            ),
            BACKFILL_KIND,
        )["points"]
        .item()
        for expected in (90.0, 45.0)
    ]
    assert docked[0] < docked[1] < 0.0


def test_conceding_records_its_intermediates() -> None:
    """The rate, clean-sheet chance and deduction are stored."""
    rows = ConcedingComponent().points(
        scored_rows(1.1), minutes_rows(), BACKFILL_KIND
    )

    diagnostics = json.loads(rows["diagnostics"].item())
    assert diagnostics["rate"] == pytest.approx(1.1)
    assert 0.0 <= diagnostics["p_clean_sheet"] <= 1.0
    assert diagnostics["expected_deduction"] > 0.0


def test_conceding_survives_a_negative_predicted_rate() -> None:
    """A regressor undershooting below zero is a certain clean sheet."""
    rows = ConcedingComponent().points(
        scored_rows(-1.0, position="MID"),
        minutes_rows(p_sixty_plus=1.0),
        BACKFILL_KIND,
    )

    assert rows["points"].item() == pytest.approx(1.0)


# --- The shared floor expectation -------------------------------------


def mass_sum_floor(rate: float, divisor: int) -> float:
    """Return E[floor(N / d)] by summing against the Poisson mass."""
    return sum((n // divisor) * poisson.pmf(n, rate) for n in range(0, 200))


# Ceilings are what an expectation reaches, not what a scoreline does:
# both models predict a conditional mean.
@pytest.mark.parametrize(
    ("divisor", "rate"),
    [(2, rate) for rate in (0.0, 0.5, 1.4, 2.5, 4.0)]
    + [(3, rate) for rate in (0.0, 0.5, 1.4, 2.8, 5.0)],
)
def test_expected_floor_matches_the_mass_function(
    divisor: int, rate: float
) -> None:
    """E[floor(N / d)] summed over the survival function equals the mass sum.

    The helper sums P(N >= d), P(N >= 2d), ... which is a different
    expression from summing ``floor(n / d)`` against the Poisson mass.
    Both divisors in use are checked: conceding's two and saves' three.
    """
    computed = expected_floor(np.array([rate]), divisor, PoissonCounts())

    assert computed[0] == pytest.approx(
        mass_sum_floor(rate, divisor), abs=1e-3
    )


def test_five_terms_is_enough_for_both_divisors() -> None:
    """Pin what truncating at ``FLOOR_TERMS`` actually costs.

    The sum is cut off at five terms, so it always under-states the
    expectation, and by more as the rate climbs towards the fifth
    multiple of the divisor. Conceding is the exposed one -- its
    divisor is smaller, so its fifth term sits at ten rather than
    fifteen -- and a thousandth of a point against a four-point clean
    sheet is why the constant stands. A third caller with a smaller
    divisor or a fatter rate should re-check this rather than assume it.
    """
    worst = {
        divisor: max(
            mass_sum_floor(rate, divisor)
            - expected_floor(np.array([rate]), divisor, PoissonCounts())[0]
            for rate in ceiling
        )
        for divisor, ceiling in ((2, (2.5, 4.0)), (3, (2.8, 5.0)))
    }

    assert 0.0 <= worst[2] < 1e-3
    assert 0.0 <= worst[3] < 1e-3


def test_expected_floor_is_zero_at_a_zero_rate() -> None:
    """Nothing expected means nothing floored."""
    assert expected_floor(np.array([0.0]), 3, PoissonCounts())[0] == 0.0


def test_expected_floor_is_monotonic_in_the_rate() -> None:
    """More expected events can never floor to fewer."""
    floored = expected_floor(
        np.array([0.5, 1.5, 3.0, 6.0]), 3, PoissonCounts()
    )

    assert list(floored) == sorted(floored)


# --- Saves ------------------------------------------------------------


def test_saves_pays_the_expected_floor_of_a_third_of_the_saves() -> None:
    """A keeper is paid E[floor(N / 3)], not E[N] / 3.

    The linear form is biased upward at every rate -- it pays for the
    two saves that earn nothing -- so the mass-function sum is what the
    points are checked against.
    """
    rate = 3.0
    rows = SavesComponent().points(
        scored_rows(rate, position="GK"),
        minutes_rows(),
        BACKFILL_KIND,
    )

    expected = sum((n // 3) * poisson.pmf(n, rate) for n in range(0, 60))
    assert rows["points"].item() == pytest.approx(expected, abs=1e-6)


def test_saves_pays_less_than_the_linear_third() -> None:
    """The floor is strictly below the linear rate over three."""
    rate = 4.0
    rows = SavesComponent().points(
        scored_rows(rate, position="GK"), minutes_rows(), BACKFILL_KIND
    )

    assert rows["points"].item() < rate / 3.0


def test_saves_scales_the_rate_by_the_minutes_expected() -> None:
    """A per-90 rate over half a match is half the exposure."""
    rate = 6.0
    rows = SavesComponent().points(
        scored_rows(rate, position="GK"),
        minutes_rows(expected_minutes=45.0),
        BACKFILL_KIND,
    )

    exposed = rate * 45.0 / 90.0
    expected = sum((n // 3) * poisson.pmf(n, exposed) for n in range(0, 60))
    assert rows["points"].item() == pytest.approx(expected, abs=1e-6)


def test_saves_pays_nothing_to_a_keeper_who_will_not_play() -> None:
    """No minutes, no exposure, no save points."""
    rows = SavesComponent().points(
        scored_rows(5.0, position="GK"),
        minutes_rows(expected_minutes=0.0),
        BACKFILL_KIND,
    )

    assert rows["points"].item() == pytest.approx(0.0)


def test_saves_survives_a_negative_predicted_rate() -> None:
    """A regressor may predict below zero; the points floor at zero."""
    rows = SavesComponent().points(
        scored_rows(-2.0, position="GK"), minutes_rows(), BACKFILL_KIND
    )

    assert rows["points"].item() == pytest.approx(0.0)


def test_saves_is_monotonic_in_the_rate() -> None:
    """A busier keeper is never paid less."""
    rows = SavesComponent().points(
        scored_rows(1.0, 3.0, 6.0, position="GK"),
        minutes_rows(n=3),
        BACKFILL_KIND,
    )

    points = rows["points"].to_list()
    assert points == sorted(points)


def test_saves_records_its_intermediates() -> None:
    """The rate and the exposed count are kept for a look at a pick."""
    rows = SavesComponent().points(
        scored_rows(3.0, position="GK"),
        minutes_rows(expected_minutes=45.0),
        BACKFILL_KIND,
    )

    diagnostics = json.loads(rows["diagnostics"].item())
    assert diagnostics["per_90"] == pytest.approx(3.0)
    assert diagnostics["expected_count"] == pytest.approx(1.5)
