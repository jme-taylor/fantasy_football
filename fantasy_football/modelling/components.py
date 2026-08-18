"""The scoring components a points prediction is summed from.

A prediction used to be one number from one model. It is now a sum of
components -- ``E[total] = sum of E[component]``, which holds by
linearity even though the components correlate. Every position is
decomposed; there is no longer a whole-prediction component, so a new
position is an entry in :data:`POSITION_COMPONENTS` naming the
components it is paid, not a model of its own.

Components live as rows in ``points_component`` rather than as a table
each, so adding one is a new :class:`Component` value and an entry in
:data:`POSITION_COMPONENTS`.

There is no residual component. Bonus points and red cards are in no
prediction at all, so every score is low by roughly a player's bonus
expectation -- concentrated in the high-BPS players, which makes it a
ranking distortion rather than a constant offset. It applies to all four
positions equally, so totals stay comparable across them.

A keeper is paid fewer components than anyone. FPL pays him no defensive
contribution, and his goals and assists are rare enough to be a mass of
exact zeros, which is why the pooled heads train on the outfield alone.
His yellow cards are modelled by nobody: the pooled head leans on fouls
committed, near-constant zero for a keeper, so it would hand back its
intercept for about a tenth of a point a match. That last one is a real
if small optimism in a keeper's score, unlike the other two.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np
import polars as pl
from numpy.typing import NDArray

from fantasy_football.constants import DEFCON_THRESHOLD_BY_POSITION
from fantasy_football.modelling.distributions import (
    CountDistribution,
    PoissonCounts,
)
from fantasy_football.storage.tables import (
    MINUTES_PREDICTION,
    POINTS_COMPONENT,
    POINTS_PREDICTION,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

#: Match-grain keys shared by every prediction frame.
KEY_COLUMNS = ["season", "gw", "element", "opponent"]

#: The column a component's model writes its output into. One name for
#: every component, so the contract does not vary with the model.
PREDICTED_VALUE = "predicted_value"

#: What the minutes model produces. Nothing here may ever appear in a
#: component's feature list -- see ``PointsComponent.model_features``.
MINUTES_OUTPUTS = ("expected_minutes", "p_zero", "p_partial", "p_sixty_plus")

#: What clearing the threshold pays.
DEFCON_POINTS = 2.0

#: What a goal pays, by position. The reason the goals model predicts a
#: rate rather than points: one rate is worth three different amounts,
#: so the conversion belongs here and not in the target.
GOALS_POINTS_BY_POSITION: dict[str, float] = {
    "GK": 6.0,
    "DEF": 6.0,
    "MID": 5.0,
    "FWD": 4.0,
}

#: What an assist pays. Flat across every position, unlike a goal, so it
#: is a scalar rather than a table -- a mapping of four identical values
#: would misstate where the variation is.
ASSIST_POINTS = 3.0

#: What a yellow card costs. Flat across every position, so a scalar
#: rather than a table, for the reason :data:`ASSIST_POINTS` is one.
YELLOW_CARD_POINTS = -1.0

#: What a clean sheet pays, by position.
CLEAN_SHEET_POINTS_BY_POSITION: dict[str, float] = {
    "GK": 4.0,
    "DEF": 4.0,
    "MID": 1.0,
    "FWD": 0.0,
}

# Held apart from the table above rather than folded into it: the two
# are different rules with different position scopes, and a single table
# would have to say "a midfielder keeps a clean sheet but is never
# docked" as a zero that reads like a price.
CONCEDING_DEDUCTION_POSITIONS: frozenset[str] = frozenset({"GK", "DEF"})

# How many terms of E[floor(N / d)] = sum over k >= 1 of P(N >= dk) are
# summed. The terms fall away fast: at a rate of three, the sixth is
# under a thousandth of a point. Checked against both divisors in use --
# conceding's two and saves' three -- so a third caller should re-check
# rather than assume.
FLOOR_TERMS = 5

#: How many goals conceded cost a point.
CONCEDING_DIVISOR = 2

#: How many saves earn a point.
SAVES_DIVISOR = 3


class Component(StrEnum):
    """One scoring component of an FPL points total."""

    #: Points for playing at all, and for playing an hour.
    APPEARANCE = "appearance"
    #: Points for clearing the defensive-contribution threshold.
    DEFCON = "defcon"
    #: Points for scoring, worth a different amount per position.
    GOALS = "goals"
    #: Points for the final pass, worth three to everyone.
    ASSISTS = "assists"
    #: The clean sheet, and the goals-conceded deduction against it.
    CONCEDING = "conceding"
    #: Points lost to bookings. Yellows only -- see YellowCardsComponent.
    YELLOW_CARDS = "yellow_cards"
    #: Points for shot-stopping, at one per three saves.
    SAVES = "saves"


# Which components each position's prediction is summed from. Read as
# configuration rather than branched on in code, so putting a position
# back to a single model is an edit here and not a revert.
#
# TODO (JT): nothing scores the composed total, so the cost of dropping
# the residual is unmeasured.
POSITION_COMPONENTS: dict[str, tuple[Component, ...]] = {
    # See the module docstring for what a keeper is not paid.
    "GK": (
        Component.APPEARANCE,
        Component.SAVES,
        Component.CONCEDING,
    ),
    "DEF": (
        Component.APPEARANCE,
        Component.DEFCON,
        Component.GOALS,
        Component.ASSISTS,
        Component.CONCEDING,
        Component.YELLOW_CARDS,
    ),
    "MID": (
        Component.APPEARANCE,
        Component.DEFCON,
        Component.GOALS,
        Component.ASSISTS,
        Component.CONCEDING,
        Component.YELLOW_CARDS,
    ),
    # No conceding: a forward is paid nothing either way.
    "FWD": (
        Component.APPEARANCE,
        Component.DEFCON,
        Component.GOALS,
        Component.ASSISTS,
        Component.YELLOW_CARDS,
    ),
}

# What identifies one fixture leg of one position's prediction.
COMPOSE_GROUP = [
    "season",
    "gw",
    "element",
    "opponent",
    "position",
    "prediction_kind",
]


@runtime_checkable
class PointsComponent(Protocol):
    """Turns one model's output into points for one scoring component.

    Minutes arrive as an argument rather than as a model feature, and
    that asymmetry is the point: every component depends on the minutes
    forecast and nothing depends on the components, so this is a
    two-layer graph rather than a DAG. Stating the dependency in the
    signature is what stops a component quietly reading a different
    minutes prediction from its neighbours -- or, worse, taking minutes
    as a feature and double-counting it.
    """

    @property
    def component(self) -> Component:
        """Return which component these points belong to."""
        ...

    @property
    def model_features(self) -> tuple[str, ...]:
        """Return the feature names this component's model reads."""
        ...

    def points(
        self,
        rows: pl.DataFrame,
        minutes: pl.DataFrame,
        kind: str,
    ) -> pl.DataFrame:
        """Turn scored rows into component rows.

        Parameters
        ----------
        rows : pl.DataFrame
            Keys, ``position`` and this component's model output in
            :data:`PREDICTED_VALUE`.
        minutes : pl.DataFrame
            Minutes-model predictions keyed the same way.
        kind : str
            Which prediction kind these rows belong to.

        Returns
        -------
        pl.DataFrame
            Rows shaped for ``points_component``.
        """
        ...


def _with_minutes(rows: pl.DataFrame, minutes: pl.DataFrame) -> pl.DataFrame:
    """Attach the minutes forecast, treating an absent one as no minutes.

    A missing minutes row must not become a null points value: a null
    survives the sum as a null and blanks out the whole composed
    prediction for that fixture leg rather than lowering it.
    """
    wanted = [
        column for column in MINUTES_OUTPUTS if column in minutes.columns
    ]
    attached = rows.join(
        minutes.select(KEY_COLUMNS + wanted), on=KEY_COLUMNS, how="left"
    )
    absent = [column for column in MINUTES_OUTPUTS if column not in wanted]
    if absent:
        attached = attached.with_columns(
            [pl.lit(0.0).alias(column) for column in absent]
        )
    missing = attached.select(
        pl.col("expected_minutes").is_null().sum()
    ).item()
    if missing:
        logger.warning(
            "%d rows have no minutes forecast; scoring them at zero.", missing
        )
    return attached.with_columns(
        [pl.col(column).fill_null(0.0) for column in MINUTES_OUTPUTS]
    )


def _component_rows(
    rows: pl.DataFrame,
    component: Component,
    kind: str,
    points: pl.Expr,
    diagnostics: pl.Expr,
) -> pl.DataFrame:
    """Shape scored rows for ``points_component``."""
    return rows.with_columns(
        prediction_kind=pl.lit(kind),
        component=pl.lit(str(component)),
        points=points.cast(pl.Float64),
        model_version=pl.lit(None, dtype=pl.Utf8),
        diagnostics=diagnostics,
    ).select(POINTS_COMPONENT.columns)


@dataclass(frozen=True)
class AppearanceComponent:
    """Points for turning up, read straight off the minutes model.

    No model of its own: FPL pays one point for playing and two for an
    hour, so the minutes classifier's bucket probabilities already are
    the expectation. It is stored as a component anyway, because a term
    left implicit stops the stored components summing to the stored
    total.
    """

    @property
    def component(self) -> Component:
        """Return :attr:`Component.APPEARANCE`."""
        return Component.APPEARANCE

    @property
    def model_features(self) -> tuple[str, ...]:
        """Return no features, since there is no model."""
        return ()

    def points(
        self, rows: pl.DataFrame, minutes: pl.DataFrame, kind: str
    ) -> pl.DataFrame:
        """Return ``1 * p_partial + 2 * p_sixty_plus`` per row."""
        joined = _with_minutes(rows, minutes)
        return _component_rows(
            joined,
            self.component,
            kind,
            pl.col("p_partial") + 2.0 * pl.col("p_sixty_plus"),
            pl.struct("p_partial", "p_sixty_plus").struct.json_encode(),
        )


@dataclass(frozen=True)
class DefconComponent:
    """Points for clearing the defensive-contribution threshold.

    The model behind this predicts a CBIT rate per 90 and knows nothing
    about minutes. Minutes turn that rate into an expected count for the
    match, and the distribution turns the expected count into the
    probability of clearing the threshold. Running a predicted count
    straight through the threshold would instead be badly biased for the
    defenders sitting closest to it.
    """

    threshold: int = DEFCON_THRESHOLD_BY_POSITION["DEF"]
    distribution: CountDistribution = field(default_factory=PoissonCounts)

    @property
    def component(self) -> Component:
        """Return :attr:`Component.DEFCON`."""
        return Component.DEFCON

    @property
    def model_features(self) -> tuple[str, ...]:
        """Return no minutes features, by construction."""
        return ()

    def points(
        self, rows: pl.DataFrame, minutes: pl.DataFrame, kind: str
    ) -> pl.DataFrame:
        """Return ``2 * P(CBIT >= threshold)`` per row."""
        joined = _with_minutes(rows, minutes).with_columns(
            _lambda=(
                pl.col(PREDICTED_VALUE).clip(lower_bound=0.0)
                * pl.col("expected_minutes")
                / 90.0
            )
        )
        hit = self.distribution.p_at_least(
            joined["_lambda"].to_numpy(), self.threshold
        )
        joined = joined.with_columns(_p_hit=pl.Series(hit).cast(pl.Float64))
        return _component_rows(
            joined,
            self.component,
            kind,
            DEFCON_POINTS * pl.col("_p_hit"),
            pl.struct(
                pl.col(PREDICTED_VALUE).alias("rate"),
                pl.col("_lambda").alias("lambda"),
                pl.col("_p_hit").alias("p_hit"),
            ).struct.json_encode(),
        )


def expected_floor(
    rate: NDArray[np.float64],
    divisor: int,
    distribution: CountDistribution,
) -> NDArray[np.float64]:
    """Return E[floor(N / divisor)] for each expected count.

    Two FPL rules have this shape: a point docked per two goals conceded,
    and a point paid per three saves. Neither is a survival probability
    -- a team expected to ship three loses more than one point, so
    ``P(N >= 2)`` is the wrong quantity -- but both are the sum
    P(N >= d) + P(N >= 2d) + ..., which is the same expectation written
    so that only the survival function is needed.

    Shared with the conceding head's metrics, which score that leg
    separately from the clean-sheet one. Two spellings of it would let
    the number a model is judged on drift from the number it is paid.

    Parameters
    ----------
    rate : NDArray[np.float64]
        Expected count over the exposure being priced.
    divisor : int
        How many events one point is worth.
    distribution : CountDistribution
        The count distribution the rate is read through.

    Returns
    -------
    NDArray[np.float64]
        One expected floored count per element of ``rate``.
    """
    floored = np.zeros_like(np.asarray(rate, dtype=float))
    for term in range(1, FLOOR_TERMS + 1):
        floored += distribution.p_at_least(rate, divisor * term)
    return floored


@dataclass(frozen=True)
class ConcedingComponent:
    """The clean sheet and the goals-conceded deduction, from one rate.

    The model behind this predicts goals conceded by a *team* in a
    fixture, per match rather than per 90, and the fan-out puts the same
    rate on every player of that team. Both payoffs come off that one
    rate, which is what stops a defender being read as likely to keep a
    clean sheet and likely to ship two in the same breath.

    The two legs read different exposures, deliberately. The deduction
    counts only goals conceded while the player was on, so it takes the
    rate scaled by expected minutes. The clean sheet already conditions
    on sixty minutes, and a defender who reaches sixty almost always
    finishes, so it takes the full-match rate -- scaling that leg too
    would raise the clean-sheet chance of exactly the rotation risks it
    should be lowering.

    What it does not model: goals conceded after the player is
    substituted still count against the clean-sheet leg here, where FPL
    would not count them.
    """

    distribution: CountDistribution = field(default_factory=PoissonCounts)

    @property
    def component(self) -> Component:
        """Return :attr:`Component.CONCEDING`."""
        return Component.CONCEDING

    @property
    def model_features(self) -> tuple[str, ...]:
        """Return no minutes features, by construction."""
        return ()

    def points(
        self, rows: pl.DataFrame, minutes: pl.DataFrame, kind: str
    ) -> pl.DataFrame:
        """Return the clean sheet less the deduction, for each row."""
        joined = _with_minutes(rows, minutes)
        unpriced = sorted(
            set(joined["position"].to_list())
            - set(CLEAN_SHEET_POINTS_BY_POSITION)
        )
        if unpriced:
            raise ValueError(
                f"{self.component} rows carry positions with no clean sheet "
                f"value: {unpriced}. A missing price would silently score "
                "them null and blank the whole composed prediction."
            )
        rate = np.clip(joined[PREDICTED_VALUE].to_numpy(), 0.0, None)
        exposed = rate * joined["expected_minutes"].to_numpy() / 90.0
        clean = 1.0 - self.distribution.p_at_least(rate, 1)
        deduction = expected_floor(
            exposed, CONCEDING_DIVISOR, self.distribution
        )
        price = pl.col("position").replace_strict(
            CLEAN_SHEET_POINTS_BY_POSITION, return_dtype=pl.Float64
        )
        docked = pl.col("position").is_in(CONCEDING_DEDUCTION_POSITIONS)
        joined = joined.with_columns(
            _p_clean=pl.Series(clean).cast(pl.Float64),
            _clean_sheet_points=price,
            _deduction=pl.when(docked)
            .then(pl.Series(deduction).cast(pl.Float64))
            .otherwise(0.0),
        )
        return _component_rows(
            joined,
            self.component,
            kind,
            pl.col("_clean_sheet_points")
            * pl.col("p_sixty_plus")
            * pl.col("_p_clean")
            - pl.col("_deduction"),
            pl.struct(
                pl.col(PREDICTED_VALUE).alias("rate"),
                pl.col("_p_clean").alias("p_clean_sheet"),
                pl.col("_deduction").alias("expected_deduction"),
            ).struct.json_encode(),
        )


@dataclass(frozen=True)
class SavesComponent:
    """Points for shot-stopping, at one point per three saves.

    Not a :class:`RateComponent` despite starting from a per-90 rate:
    FPL pays the floor of the count over three, and a linear
    ``rate / 3`` overpays at every rate because it pays for the two
    saves that earn nothing. The floor is taken through the same
    survival-function sum the conceding deduction uses.

    The exposure is the minutes forecast, which is close to a formality
    for a keeper -- they are substituted far less often than outfielders
    -- but it is what stops a benched second choice being paid for the
    saves his club's starter will make.
    """

    distribution: CountDistribution = field(default_factory=PoissonCounts)

    @property
    def component(self) -> Component:
        """Return :attr:`Component.SAVES`."""
        return Component.SAVES

    @property
    def model_features(self) -> tuple[str, ...]:
        """Return no minutes features, by construction."""
        return ()

    def points(
        self, rows: pl.DataFrame, minutes: pl.DataFrame, kind: str
    ) -> pl.DataFrame:
        """Return ``E[floor(saves / 3)]`` over the minutes expected."""
        joined = _with_minutes(rows, minutes).with_columns(
            _expected=(
                pl.col(PREDICTED_VALUE).clip(lower_bound=0.0)
                * pl.col("expected_minutes")
                / 90.0
            )
        )
        paid = expected_floor(
            joined["_expected"].to_numpy(), SAVES_DIVISOR, self.distribution
        )
        joined = joined.with_columns(_points=pl.Series(paid).cast(pl.Float64))
        return _component_rows(
            joined,
            self.component,
            kind,
            pl.col("_points"),
            pl.struct(
                pl.col(PREDICTED_VALUE).alias("per_90"),
                pl.col("expected_minutes"),
                pl.col("_expected").alias("expected_count"),
            ).struct.json_encode(),
        )


@dataclass(frozen=True)
class RateComponent:
    """Points from a per-90 rate, scaled by the minutes forecast.

    The shape every remaining component takes once appearance and defcon
    -- both of which are step functions of minutes rather than rates --
    have been carved out.

    ``points_per_event`` is what separates a rate of *points* from a
    rate of *events*. A model already predicting points leaves this None
    and the scaled rate is the answer. A goals model predicts goals,
    which are worth six to a defender and four to a forward, so the
    conversion happens here -- which is what lets one model serve every
    position it was trained on. An assist pays the same everywhere, so
    that model passes a scalar: the mapping form exists to express
    variation by position, and there is none.
    """

    component_name: Component
    points_per_event: Mapping[str, float] | float | None = None
    distribution: CountDistribution = field(default_factory=PoissonCounts)

    @property
    def component(self) -> Component:
        """Return the component this rate is stored as."""
        return self.component_name

    @property
    def model_features(self) -> tuple[str, ...]:
        """Return no minutes features, by construction."""
        return ()

    def points(
        self, rows: pl.DataFrame, minutes: pl.DataFrame, kind: str
    ) -> pl.DataFrame:
        """Return the per-90 rate scaled to the minutes expected."""
        joined = _with_minutes(rows, minutes).with_columns(
            _expected=(
                pl.col(PREDICTED_VALUE).clip(lower_bound=0.0)
                * pl.col("expected_minutes")
                / 90.0
            )
        )
        if self.points_per_event is None:
            return _component_rows(
                joined,
                self.component,
                kind,
                pl.col(PREDICTED_VALUE) * pl.col("expected_minutes") / 90.0,
                pl.struct(
                    pl.col(PREDICTED_VALUE).alias("per_90"),
                    pl.col("expected_minutes"),
                ).struct.json_encode(),
            )
        if isinstance(self.points_per_event, Mapping):
            unpriced = sorted(
                set(joined["position"].to_list()) - set(self.points_per_event)
            )
            if unpriced:
                raise ValueError(
                    f"{self.component} rows carry positions with no points "
                    f"value: {unpriced}. A missing price would silently "
                    "score them null and blank the whole composed "
                    "prediction."
                )
            price = pl.col("position").replace_strict(
                dict(self.points_per_event), return_dtype=pl.Float64
            )
        else:
            price = pl.lit(float(self.points_per_event), dtype=pl.Float64)
        # P(at least one) is not what the points are built from -- they
        # are linear in the count, so the expectation is enough -- but it
        # is the number wanted when a captaincy pick looks wrong.
        scored = self.distribution.p_at_least(
            joined["_expected"].to_numpy(), 1
        )
        joined = joined.with_columns(
            _p_scored=pl.Series(scored).cast(pl.Float64),
            _points_per_event=price,
        )
        return _component_rows(
            joined,
            self.component,
            kind,
            pl.col("_points_per_event") * pl.col("_expected"),
            pl.struct(
                pl.col(PREDICTED_VALUE).alias("per_90"),
                pl.col("expected_minutes"),
                pl.col("_expected").alias("expected_count"),
                pl.col("_p_scored").alias("p_scored"),
            ).struct.json_encode(),
        )


@dataclass(frozen=True)
class YellowCardsComponent:
    """Points lost to bookings, from a per-90 yellow-card rate.

    Reds are deliberately not here. They pay minus three against a
    yellow's minus one and arrive roughly forty times less often -- 44
    across the league in 2025-26 against 1422 yellows -- which is too
    few to estimate a player rate from, so they are in no component at
    all. Naming this one for yellows is what keeps reds a separate
    component when they earn one, rather than a nullable field in here.

    The conversion is a scaled rate at minus one a card, which is
    :class:`RateComponent`'s job, so that is what does the arithmetic.
    The wrapper exists for the seam: a booking is not quite a linear
    per-90 event -- late tactical fouls do not scale with minutes the
    way tackles do -- and when that is modelled properly the change
    belongs here rather than in the shared rate component every other
    head uses.
    """

    rate: "RateComponent" = field(
        default_factory=lambda: RateComponent(
            Component.YELLOW_CARDS, points_per_event=YELLOW_CARD_POINTS
        )
    )

    @property
    def component(self) -> Component:
        """Return the component these points are stored as."""
        return Component.YELLOW_CARDS

    @property
    def model_features(self) -> tuple[str, ...]:
        """Return no minutes features, by construction."""
        return ()

    def points(
        self, rows: pl.DataFrame, minutes: pl.DataFrame, kind: str
    ) -> pl.DataFrame:
        """Return the booking rate scaled to the minutes expected."""
        return self.rate.points(rows, minutes, kind)


def composite_version(versions: Mapping[Component, str]) -> str:
    """Fold each component's model version into one version string.

    A single component keeps its bare version. Anything else is named and
    sorted, so the string is stable across runs and the backfill rebuild
    check compares like with like.

    Parameters
    ----------
    versions : Mapping[Component, str]
        The model version behind each component of one prediction.

    Returns
    -------
    str
        The version stamped on the composed row.

    Raises
    ------
    ValueError
        If no components were given.
    """
    if not versions:
        raise ValueError("Cannot version a prediction with no components.")
    if len(versions) == 1:
        return next(iter(versions.values()))
    return "|".join(
        f"{component}={versions[component]}"
        for component in sorted(versions, key=str)
    )


def _reject_unknown_components(rows: pl.DataFrame) -> None:
    """Raise if any row names a component that is not declared."""
    known = {str(component) for component in Component}
    unknown = sorted(set(rows["component"].to_list()) - known)
    if unknown:
        raise ValueError(
            f"Component rows name components that do not exist: {unknown}. "
            "Add them to Component, or fix whatever wrote them -- an "
            "undeclared component is dropped from the sum silently."
        )


def _reject_duplicate_components(rows: pl.DataFrame) -> None:
    """Raise if one fixture leg carries a component twice."""
    duplicated = (
        rows.group_by(COMPOSE_GROUP + ["component"])
        .len()
        .filter(pl.col("len") > 1)
    )
    if not duplicated.is_empty():
        raise ValueError(
            f"{duplicated.height} fixture legs carry a component more than "
            "once, so the sum would double-count it. First offender: "
            f"{duplicated.row(0, named=True)}"
        )


def _drop_incomplete_predictions(
    rows: pl.DataFrame, expected: Mapping[str, Sequence[Component]]
) -> pl.DataFrame:
    """Return only the fixture legs carrying every component they need.

    Components do not all cover the same rows. The defcon head has no
    CBIT to rate before 2024-25 and neither head rates an appearance
    under fifteen minutes, so a leg can legitimately have some of its
    components and not others -- a decomposed position simply has no
    prediction for those legs.

    Summing a subset would under-predict, and the optimiser would act on
    the low number without complaint. Dropping the leg outright instead
    leaves no prediction at all, which the optimiser reads as zero and
    passes over. An absent row is safe in a way a low row is not.
    """
    positions = set(rows["position"].to_list())
    undeclared = sorted(positions - set(expected))
    if undeclared:
        raise ValueError(
            f"Component rows carry positions with no declared components: "
            f"{undeclared}. Add them to POSITION_COMPONENTS."
        )
    present = rows.group_by(COMPOSE_GROUP).agg(
        pl.col("component").alias("components")
    )
    incomplete = []
    missing_counts: dict[str, int] = {}
    for row in present.iter_rows(named=True):
        wanted = {str(component) for component in expected[row["position"]]}
        found = set(row["components"])
        # A component left behind by an earlier shape of the pipeline --
        # the monolithic rows for a position since decomposed, most
        # likely -- would be summed in on top of the components that
        # replaced it, silently doubling the prediction. That is a fault
        # in what was written, so it stops the run.
        stale = sorted(found - wanted)
        if stale:
            raise ValueError(
                f"{row['position']} carries components {stale} that it no "
                f"longer declares, so they would be summed on top of "
                f"{sorted(wanted)}. Delete them from points_component, or "
                "add them back to POSITION_COMPONENTS."
            )
        missing = wanted - found
        if missing:
            incomplete.append({key: row[key] for key in COMPOSE_GROUP})
            for component in missing:
                missing_counts[component] = (
                    missing_counts.get(component, 0) + 1
                )
    if not incomplete:
        return rows
    dropped = pl.DataFrame(incomplete)
    # Naming the component is what turns "the defenders vanished" into a
    # message. Declaring a new component drops every existing leg of that
    # position until its backfill has run, and the optimiser reads the
    # composed table, so the cause has to be in the log.
    by_component = ", ".join(
        f"{component} ({count} legs)"
        for component, count in sorted(missing_counts.items())
    )
    logger.warning(
        "Dropping %d fixture legs missing at least one declared component; "
        "they get no composed prediction rather than a partial one. "
        "Missing: %s. First: %s",
        dropped.height,
        by_component,
        dropped.row(0, named=True),
    )
    return rows.join(dropped, on=COMPOSE_GROUP, how="anti")


def compose(
    component_rows: pl.DataFrame,
    expected: Mapping[str, Sequence[Component]] | None = None,
) -> pl.DataFrame:
    """Sum component rows into one points prediction per fixture leg.

    Pure: no connection, no I/O. The only writer into
    ``points_prediction``, so "the components sum to the stored total" is
    a property of this function alone.

    Parameters
    ----------
    component_rows : pl.DataFrame
        Rows shaped like ``points_component``.
    expected : Mapping[str, Sequence[Component]] | None, optional
        Which components each position must supply. When given, legs
        missing one are dropped rather than partially summed. Defaults to
        None, meaning completeness is not checked.

    Returns
    -------
    pl.DataFrame
        Rows shaped like ``points_prediction``.

    Raises
    ------
    ValueError
        If a component is unknown, carried twice for one fixture leg, or
        carried by a position that no longer declares it.
    """
    if component_rows.is_empty():
        return POINTS_PREDICTION.coerce(
            pl.DataFrame(schema=POINTS_PREDICTION.schema)
        )
    rows = POINTS_COMPONENT.coerce(component_rows)
    _reject_unknown_components(rows)
    _reject_duplicate_components(rows)
    if expected is not None:
        rows = _drop_incomplete_predictions(rows, expected)
        if rows.is_empty():
            return POINTS_PREDICTION.coerce(
                pl.DataFrame(schema=POINTS_PREDICTION.schema)
            )
    composed = (
        rows.with_columns(
            _label=pl.format(
                "{}={}", pl.col("component"), pl.col("model_version")
            )
        )
        .group_by(COMPOSE_GROUP)
        .agg(
            pl.col("points").sum().alias("predicted_points"),
            pl.len().alias("_components"),
            pl.col("model_version").first().alias("_only_version"),
            pl.col("_label").sort().str.join("|").alias("_labels"),
        )
        .with_columns(
            model_version=pl.when(pl.col("_components") == 1)
            .then(pl.col("_only_version"))
            .otherwise(pl.col("_labels"))
        )
    )
    return POINTS_PREDICTION.coerce(composed)


def write_appearance_components(
    connection: "DuckDBPyConnection",
    kind: str,
    expected: Mapping[str, Sequence[Component]] | None = None,
) -> None:
    """Store appearance points for every position that declares them.

    There is no model here. FPL pays one point for playing and two for an
    hour, so the minutes classifier's bucket probabilities already are
    the expectation -- which is why the version stamped on these rows is
    the minutes model's.

    The rows written are the ones the position's *other* components
    already cover, not every row the minutes model scored. Those
    components are narrower -- there is no CBIT to rate before 2024-25,
    and neither head rates an appearance under fifteen minutes -- so
    keying off the minutes model instead would emit appearance points for
    legs that can never be composed. Reading the stored components also
    inherits the forward freeze for free: a leg below the first unplayed
    gameweek keeps the rows it was frozen with, so its appearance points
    are not quietly refreshed against a newer minutes model.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection.
    kind : str
        Which minutes prediction kind to read and stamp.
    expected : Mapping[str, Sequence[Component]] | None, optional
        Component declarations. Defaults to :data:`POSITION_COMPONENTS`.
    """
    declarations = POSITION_COMPONENTS if expected is None else expected
    positions = [
        position
        for position, components in declarations.items()
        if Component.APPEARANCE in components
    ]
    if not positions:
        return
    minutes = MINUTES_PREDICTION.load(connection).filter(
        pl.col("prediction_kind") == kind
    )
    if minutes.is_empty():
        logger.info("No %s minutes predictions; no appearance points.", kind)
        return
    stored = POINTS_COMPONENT.load(connection).filter(
        (pl.col("prediction_kind") == kind)
        & (pl.col("component") != str(Component.APPEARANCE))
        & (pl.col("position").is_in(positions))
    )
    keyed = (
        stored.select(KEY_COLUMNS + ["position"])
        .unique()
        .join(
            minutes.select(KEY_COLUMNS + ["model_version"]),
            on=KEY_COLUMNS,
            how="inner",
        )
    )
    if keyed.is_empty():
        logger.info("No %s components at a decomposed position.", kind)
        return
    rows = AppearanceComponent().points(
        keyed.drop("model_version"), minutes, kind
    )
    rows = rows.drop("model_version").join(
        keyed.select(KEY_COLUMNS + ["model_version"]),
        on=KEY_COLUMNS,
        how="left",
    )
    for (season,), partition in rows.group_by(["season"]):
        POINTS_COMPONENT.replace_partition(
            connection,
            partition,
            equals={
                "season": season,
                "prediction_kind": kind,
                "component": str(Component.APPEARANCE),
            },
        )


def compose_points(connection: "DuckDBPyConnection") -> None:
    """Rebuild ``points_prediction`` from the stored components.

    The only writer into ``points_prediction``. ``points_component`` is
    the source of truth, including the frozen forward rows below the
    first unplayed gameweek: those rows survive in the component table,
    so recomposing the whole partition reproduces them rather than
    dropping them.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection holding both tables.
    """
    components = POINTS_COMPONENT.load(connection)
    if components.is_empty():
        logger.info("No stored components; leaving points_prediction alone.")
        return
    composed = compose(components, expected=POSITION_COMPONENTS)
    for (season, kind), partition in composed.group_by(
        ["season", "prediction_kind"]
    ):
        POINTS_PREDICTION.replace_partition(
            connection,
            partition,
            equals={"season": season, "prediction_kind": kind},
        )
    logger.info(
        "Composed %d points_prediction rows from %d component rows.",
        composed.height,
        components.height,
    )
