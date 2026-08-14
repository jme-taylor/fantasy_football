"""The three scoring components a decomposed DEF prediction is built from.

Every component is tested against the same contract: given keyed rows
carrying its model's output and a frame of minutes predictions, it
returns component rows whose points can be hand-computed.
"""

import json

import polars as pl
import pytest

from fantasy_football.modelling.components import (
    MINUTES_OUTPUTS,
    POSITION_COMPONENTS,
    PREDICTED_VALUE,
    AppearanceComponent,
    Component,
    DefconComponent,
    PointsComponent,
    RateComponent,
    TotalComponent,
)
from fantasy_football.modelling.defcon import DefconRatePredictor
from fantasy_football.modelling.defender import (
    DefenderPointsPredictor,
    DefenderResidualPointsPredictor,
)
from fantasy_football.modelling.distributions import PoissonCounts
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
    [AppearanceComponent(), defcon(), RateComponent(Component.RESIDUAL)],
    ids=lambda c: str(c.component),
)
def test_every_component_satisfies_the_protocol(component) -> None:
    """Each implementation is a PointsComponent."""
    assert isinstance(component, PointsComponent)


@pytest.mark.parametrize(
    "component",
    [AppearanceComponent(), defcon(), RateComponent(Component.RESIDUAL)],
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
    [AppearanceComponent(), defcon(), RateComponent(Component.RESIDUAL)],
    ids=lambda c: str(c.component),
)
def test_no_component_reads_a_minutes_feature(component) -> None:
    """Minutes reach a component as an argument, never as a feature.

    The load-bearing rule of the decomposition: minutes are applied once,
    at composition. A component that took a minutes column as a model
    feature would double-count it the moment a second component was
    added, and the arithmetic would stop being a decomposition without
    anything failing.
    """
    assert not set(component.model_features) & set(MINUTES_OUTPUTS)


def test_the_undecomposed_component_is_the_stated_exception() -> None:
    """TOTAL declares the minutes it reads rather than hiding them.

    An undecomposed model still takes expected_minutes as a feature, and
    that is fine because nothing else scales its output. Declaring it
    keeps the rule above meaningful: the exception is written down rather
    than silently passing because the component happens to list nothing.
    """
    assert set(TotalComponent().model_features) == set(MINUTES_OUTPUTS)


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


def test_rate_component_scales_a_per_90_rate_by_minutes() -> None:
    """Per-90 output becomes per-match by the minutes played."""
    rows = RateComponent(Component.RESIDUAL).points(
        scored_rows(4.0), minutes_rows(expected_minutes=45.0), BACKFILL_KIND
    )

    assert rows["points"].item() == pytest.approx(2.0)


def test_rate_component_passes_negative_points_through() -> None:
    """A card-heavy defender's residual is allowed to be negative."""
    rows = RateComponent(Component.RESIDUAL).points(
        scored_rows(-1.0), minutes_rows(expected_minutes=90.0), BACKFILL_KIND
    )

    assert rows["points"].item() == pytest.approx(-1.0)


# --- Missing minutes -------------------------------------------------


@pytest.mark.parametrize(
    "component",
    [AppearanceComponent(), defcon(), RateComponent(Component.RESIDUAL)],
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
    [DefconRatePredictor, DefenderResidualPointsPredictor],
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


def test_every_decomposed_def_component_is_declared() -> None:
    """DEF's declared components are exactly the ones with a writer."""
    assert set(POSITION_COMPONENTS["DEF"]) == {
        Component.APPEARANCE,
        Component.DEFCON,
        Component.RESIDUAL,
    }


def test_the_residual_model_is_the_monolith_minus_minutes() -> None:
    """Nothing replaces the dropped minutes features.

    A stand-in for expected_minutes would put the double-count back in
    through the side door, so the residual list must be a strict subset.
    """
    assert set(DefenderResidualPointsPredictor.FEATURES) == (
        set(DefenderPointsPredictor.FEATURES) - set(MINUTES_OUTPUTS)
    )
