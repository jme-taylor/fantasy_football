"""The component ablation: oracle pricing, leg scoping and the deltas."""

import polars as pl
import pytest

from fantasy_football.modelling.ablation import (
    BASELINE,
    MINUTES_VARIANT,
    Head,
    LoadedHead,
    _paid_positions,
    _with_deltas,
    build_universe,
    leverage_table,
    oracle_minutes,
    realised_points,
)
from fantasy_football.modelling.components import (
    KEY_COLUMNS,
    AppearanceComponent,
    Component,
)

SEASON = "2025-26"


def legs(**columns: object) -> pl.DataFrame:
    """Return fixture legs carrying whatever the pricing under test reads."""
    height = max(
        len(value) for value in columns.values() if isinstance(value, list)
    )
    base = {
        "season": [SEASON] * height,
        "gw": list(range(1, height + 1)),
        "element": [1] * height,
        "opponent": [2] * height,
    }
    return pl.DataFrame({**base, **columns})


def priced(component: Component, frame: pl.DataFrame) -> list[float]:
    """Return the realised points for a component, one per leg."""
    return (
        frame.select(realised_points(component, frame)).to_series().to_list()
    )


class TestRealisedPoints:
    """What each component actually paid, given what happened."""

    def test_appearance_pays_for_the_hour(self) -> None:
        """Nothing for a no-show, one for a cameo, two for an hour."""
        frame = legs(minutes=[0, 45, 59, 60, 90], position=["MID"] * 5)
        assert priced(Component.APPEARANCE, frame) == [0.0, 1.0, 1.0, 2.0, 2.0]

    def test_goals_are_worth_what_the_position_pays(self) -> None:
        """One goal each, priced at six for a defender and four for a forward."""
        frame = legs(
            minutes=[90, 90],
            position=["DEF", "FWD"],
            actual_rate=[1.0, 1.0],
        )
        assert priced(Component.GOALS, frame) == [6.0, 4.0]

    def test_a_rate_over_part_of_a_match_recovers_its_count(self) -> None:
        """A per-90 rate is undone by the minutes it was built from."""
        # Two goals in 45 minutes is a rate of four per 90.
        frame = legs(minutes=[45], position=["FWD"], actual_rate=[4.0])
        assert priced(Component.GOALS, frame) == [8.0]

    def test_assists_and_cards_are_flat(self) -> None:
        """Three a pass, minus one a booking, at every position."""
        frame = legs(minutes=[90], position=["MID"], actual_rate=[2.0])
        assert priced(Component.ASSISTS, frame) == [6.0]
        assert priced(Component.YELLOW_CARDS, frame) == [-2.0]

    def test_saves_pay_per_three(self) -> None:
        """Two saves pay nothing; the third pays the point."""
        frame = legs(
            minutes=[90, 90, 90],
            position=["GK"] * 3,
            actual_rate=[2.0, 3.0, 5.0],
        )
        assert priced(Component.SAVES, frame) == [0.0, 1.0, 1.0]

    def test_defcon_is_a_threshold_not_a_rate(self) -> None:
        """A defender clears at ten, a midfielder at twelve."""
        frame = legs(
            minutes=[90, 90, 90],
            position=["DEF", "MID", "MID"],
            actual_rate=[10.0, 10.0, 12.0],
        )
        assert priced(Component.DEFCON, frame) == [2.0, 0.0, 2.0]

    def test_conceding_pays_the_sheet_and_docks_the_goals(self) -> None:
        """The sheet needs the hour; the deduction spares midfielders."""
        frame = legs(
            minutes=[90, 45, 90, 90],
            position=["DEF", "DEF", "DEF", "MID"],
            actual_rate=[0.0, 0.0, 2.0, 2.0],
        )
        assert priced(Component.CONCEDING, frame) == [4.0, 0.0, -1.0, 0.0]


class TestOracleMinutes:
    """A perfect forecast is a one-hot of the bucket that happened."""

    def test_buckets_collapse_onto_what_happened(self) -> None:
        """Each leg puts all its probability on its realised bucket."""
        frame = oracle_minutes(legs(minutes=[0, 30, 90]))
        assert frame["p_zero"].to_list() == [1.0, 0.0, 0.0]
        assert frame["p_partial"].to_list() == [0.0, 1.0, 0.0]
        assert frame["p_sixty_plus"].to_list() == [0.0, 0.0, 1.0]
        assert frame["expected_minutes"].to_list() == [0.0, 30.0, 90.0]


class TestPaidPositions:
    """Which positions a component is actually summed into."""

    def test_conceding_skips_the_forward(self) -> None:
        """A forward is paid nothing either way, so he carries no leg."""
        assert "FWD" not in _paid_positions(Component.CONCEDING)
        assert set(_paid_positions(Component.CONCEDING)) == {
            "GK",
            "DEF",
            "MID",
        }

    def test_saves_are_the_keeper_alone(self) -> None:
        """Only a keeper is paid for shot-stopping."""
        assert _paid_positions(Component.SAVES) == ["GK"]


def head_rows(component: Component, gws: list[int]) -> LoadedHead:
    """Return a head covering one leg per gameweek."""
    return LoadedHead(
        Head("experiment", component, AppearanceComponent()),
        pl.DataFrame(
            {
                "season": [SEASON] * len(gws),
                "gw": gws,
                "element": [1] * len(gws),
                "opponent": [2] * len(gws),
                "position": ["FWD"] * len(gws),
                "predicted_rate": [0.5] * len(gws),
                "actual_rate": [1.0] * len(gws),
            }
        ),
    )


class TestBuildUniverse:
    """Which legs every variant can be scored on."""

    def test_a_leg_missing_one_component_is_out(self) -> None:
        """A forward needs all five of his; four is no prediction at all."""
        heads = [
            head_rows(Component.GOALS, [1, 2]),
            head_rows(Component.ASSISTS, [1, 2]),
            head_rows(Component.DEFCON, [1, 2]),
            # The booking head covers only the first gameweek.
            head_rows(Component.YELLOW_CARDS, [1]),
        ]
        covered = legs(minutes=[90, 90])
        universe = build_universe(heads, covered, covered.select(KEY_COLUMNS))
        assert universe["gw"].to_list() == [1]

    def test_a_leg_with_no_actuals_is_out(self) -> None:
        """A leg nothing realised cannot be scored either way."""
        heads = [
            head_rows(component, [1, 2])
            for component in (
                Component.GOALS,
                Component.ASSISTS,
                Component.DEFCON,
                Component.YELLOW_CARDS,
            )
        ]
        actuals = legs(minutes=[90]).head(1)
        minutes = legs(minutes=[90, 90]).select(KEY_COLUMNS)
        universe = build_universe(heads, actuals, minutes)
        assert universe["gw"].to_list() == [1]


def scored(variant: str, position: str, **metrics: float) -> dict[str, object]:
    """Return one scored row."""
    return {
        "variant": variant,
        "position": position,
        "legs": 10,
        "mae": metrics.get("mae", 1.0),
        "rmse": 2.0,
        "spearman": metrics.get("spearman", 0.1),
        "precision_at_k": metrics.get("precision_at_k", 0.2),
    }


class TestDeltas:
    """Every variant is reported against the baseline for its position."""

    def test_deltas_are_taken_against_the_position_baseline(self) -> None:
        """A variant's movement is its own position's, not another's."""
        results = pl.DataFrame(
            [
                scored(BASELINE, "MID", mae=1.0, precision_at_k=0.2),
                scored("goals", "MID", mae=0.6, precision_at_k=0.5),
                scored(BASELINE, "DEF", mae=2.0, precision_at_k=0.4),
                scored("goals", "DEF", mae=1.5, precision_at_k=0.45),
            ]
        )
        deltas = _with_deltas(results)
        mid = deltas.filter(
            (pl.col("position") == "MID") & (pl.col("variant") == "goals")
        )
        assert mid["delta_mae"].item() == pytest.approx(-0.4)
        assert mid["delta_precision_at_k"].item() == pytest.approx(0.3)
        defenders = deltas.filter(
            (pl.col("position") == "DEF") & (pl.col("variant") == "goals")
        )
        assert defenders["delta_precision_at_k"].item() == pytest.approx(0.05)

    def test_a_component_a_position_is_not_paid_is_not_reported(self) -> None:
        """A row of zeros would read as measured indifference."""
        results = _with_deltas(
            pl.DataFrame(
                [
                    scored(BASELINE, "FWD"),
                    scored(MINUTES_VARIANT, "FWD"),
                    scored(str(Component.GOALS), "FWD"),
                    scored(str(Component.SAVES), "FWD"),
                ]
            )
        )
        reported = set(leverage_table(results)["variant"].to_list())
        assert str(Component.SAVES) not in reported
        assert {BASELINE, MINUTES_VARIANT, str(Component.GOALS)} <= reported
