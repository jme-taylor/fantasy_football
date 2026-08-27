"""Which component's error costs the composed prediction the most.

One component at a time is replaced by what actually happened, every other
component is left at its out-of-fold prediction, and the composed total is
rescored. The drop in error is that component's *leverage*, which is not the
same as its error: a large error on a component nothing ranks on costs
nothing, and a small one on a component that separates the top of a gameweek
costs a lot.

Reads the evaluation tables written by ``store_fold_predictions`` -- held-out
rows, not the in-sample backfill -- so nothing here trains, fits or writes.

Minutes gets a variant of its own rather than a component slot. It multiplies
every rate component as well as paying the appearance points, so "perfect
minutes" is the ceiling on everything downstream of it.
"""

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mlflow
import mlflow.tracking
import polars as pl
from mlflow.exceptions import MlflowException

from fantasy_football.constants import (
    DEFCON_THRESHOLD_BY_POSITION,
    MLFLOW_TRACKING_URI,
    PRECISION_K_BY_POSITION,
)
from fantasy_football.modelling.assists import (
    EXPERIMENT_NAME as ASSISTS_EXPERIMENT,
)
from fantasy_football.modelling.assists import AssistsRatePredictor
from fantasy_football.modelling.components import (
    ASSIST_POINTS,
    CLEAN_SHEET_POINTS_BY_POSITION,
    CONCEDING_DEDUCTION_POSITIONS,
    CONCEDING_DIVISOR,
    DEFCON_POINTS,
    GOALS_POINTS_BY_POSITION,
    KEY_COLUMNS,
    POSITION_COMPONENTS,
    PREDICTED_VALUE,
    SAVES_DIVISOR,
    YELLOW_CARD_POINTS,
    AppearanceComponent,
    Component,
    PointsComponent,
    compose,
)
from fantasy_football.modelling.conceding import (
    EXPERIMENT_NAME as CONCEDING_EXPERIMENT,
)
from fantasy_football.modelling.conceding import ConcedingPredictor
from fantasy_football.modelling.defcon import (
    CbirtRatePredictor,
    DefconRatePredictor,
)
from fantasy_football.modelling.goals import (
    EXPERIMENT_NAME as GOALS_EXPERIMENT,
)
from fantasy_football.modelling.goals import GoalsRatePredictor
from fantasy_football.modelling.metrics import (
    mae,
    precision_at_k,
    rmse,
    spearman_by_gw,
)
from fantasy_football.modelling.saves import (
    EXPERIMENT_NAME as SAVES_EXPERIMENT,
)
from fantasy_football.modelling.saves import SavesRatePredictor
from fantasy_football.modelling.yellow_cards import (
    EXPERIMENT_NAME as YELLOW_CARDS_EXPERIMENT,
)
from fantasy_football.modelling.yellow_cards import YellowCardsRatePredictor
from fantasy_football.storage.tables import (
    POINTS_COMPONENT,
    TEAM_FIXTURE,
    TEST_CONCEDING_PREDICTION,
    TEST_MINUTES_PREDICTION,
    TEST_POINTS_PREDICTION,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# The experiment names the defcon heads are trained under. Unlike every
# other head these live in the training scripts rather than the module,
# so they are spelled out here rather than imported.
DEFCON_EXPERIMENT = "def-defcon-rate-model"
CBIRT_EXPERIMENT = "mid-fwd-cbirt-rate-model"

# Stamped on the component rows this builds. Never stored -- it exists so
# the frames satisfy the points_component shape that compose() reads.
ABLATION_KIND = "ablation"

# The variant that changes no component. Everything else is reported
# against it.
BASELINE = "baseline"

# The variant that replaces the minutes forecast rather than a component.
MINUTES_VARIANT = "minutes"

# What the ranking metrics group on. A holdout spanning seasons would
# otherwise rank two years' players against each other.
RANKING_GROUP = ("season", "gw")


@dataclass(frozen=True)
class Head:
    """One trained model feeding one component.

    Attributes
    ----------
    experiment : str
        The MLflow experiment its runs are logged under, which is how a
        run_id in the shared evaluation table is traced back to a head.
    component : Component
        The component its output is paid as. Two heads can share one --
        defcon is modelled separately for defenders and for everyone
        else -- so this is not a key.
    impl : PointsComponent
        The same component implementation serving uses, so the predicted
        side of the ablation is the arithmetic in production and not a
        second spelling of it.
    """

    experiment: str
    component: Component
    impl: PointsComponent


#: The per-90 rate heads, all of which write into the shared points
#: evaluation table and are told apart by their run's experiment.
RATE_HEADS: tuple[Head, ...] = (
    Head(GOALS_EXPERIMENT, Component.GOALS, GoalsRatePredictor.COMPONENT_IMPL),
    Head(
        ASSISTS_EXPERIMENT,
        Component.ASSISTS,
        AssistsRatePredictor.COMPONENT_IMPL,
    ),
    Head(SAVES_EXPERIMENT, Component.SAVES, SavesRatePredictor.COMPONENT_IMPL),
    Head(
        YELLOW_CARDS_EXPERIMENT,
        Component.YELLOW_CARDS,
        YellowCardsRatePredictor.COMPONENT_IMPL,
    ),
    Head(
        DEFCON_EXPERIMENT,
        Component.DEFCON,
        DefconRatePredictor.COMPONENT_IMPL,
    ),
    Head(
        CBIRT_EXPERIMENT, Component.DEFCON, CbirtRatePredictor.COMPONENT_IMPL
    ),
)

#: The team-grain head. Separate because its evaluation rows are one per
#: team-fixture and have to be fanned out to the players who share them.
CONCEDING_HEAD = Head(
    CONCEDING_EXPERIMENT,
    Component.CONCEDING,
    ConcedingPredictor.COMPONENT_IMPL,
)


def latest_run_ids(
    run_ids: Sequence[str], tracking_uri: str = MLFLOW_TRACKING_URI
) -> dict[str, str]:
    """Return the newest run per experiment among ``run_ids``.

    The evaluation tables carry every run ever scored, and the run_id is
    the only thing telling one head's rows from another's, so the head a
    run belongs to is read back off MLflow rather than stored.

    Parameters
    ----------
    run_ids : Sequence[str]
        Run ids present in an evaluation table.
    tracking_uri : str, optional
        MLflow tracking store. Defaults to the project's.

    Returns
    -------
    dict[str, str]
        Experiment name to its newest run id. Runs MLflow no longer
        knows about are skipped with a warning.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    newest: dict[str, tuple[int, str]] = {}
    for run_id in run_ids:
        try:
            run = client.get_run(run_id)
            experiment = client.get_experiment(run.info.experiment_id).name
        except MlflowException:
            logger.warning(
                "Run %s is in an evaluation table but not in MLflow; "
                "skipping it.",
                run_id,
            )
            continue
        started = run.info.start_time or 0
        if experiment not in newest or started > newest[experiment][0]:
            newest[experiment] = (started, run_id)
    return {name: run for name, (_, run) in newest.items()}


def load_actuals(connection: "DuckDBPyConnection") -> pl.DataFrame:
    """Return realised minutes and points for every fixture leg.

    ``player_match`` rather than the minutes evaluation table: the two
    agree on the rows they share, and this one covers every leg a head
    was scored on rather than only the minutes model's folds.
    """
    return connection.sql(
        "SELECT season, gw, element, opponent, is_home, "
        "minutes, total_points AS actual_points FROM player_match"
    ).pl()


def load_leg_teams(connection: "DuckDBPyConnection") -> pl.DataFrame:
    """Return each fixture leg's club and its opposition, both by name.

    The conceding head is keyed by club name and the player heads by FPL
    element and opponent id, so the two are joined through the club a
    player turned out for and the fixture that club played. The kickoff
    is what picks the fixture: a double gameweek can put a club at home
    twice or away twice, so neither the gameweek nor ``is_home`` tells
    its two legs apart, and joining on either fans the player across
    both.
    """
    legs = connection.sql(
        "SELECT m.season, m.gw, m.element, m.opponent, m.kickoff_time, "
        "w.team FROM player_match m "
        "JOIN player_week w USING (season, gw, element)"
    ).pl()
    fixtures = TEAM_FIXTURE.load(connection).select(
        "season", "gw", "team", "kickoff_time", "opposition"
    )
    return legs.join(
        fixtures, on=["season", "gw", "team", "kickoff_time"], how="inner"
    ).drop("kickoff_time")


@dataclass(frozen=True)
class LoadedHead:
    """One head's held-out rows, ready to be scored either way.

    Attributes
    ----------
    head : Head
        Which head these rows came from.
    rows : pl.DataFrame
        Keys, ``position``, the predicted per-90 rate and the realised
        one, plus ``conceded`` for the team-grain head.
    """

    head: Head
    rows: pl.DataFrame


def load_rate_heads(
    points_rows: pl.DataFrame, runs: Mapping[str, str]
) -> list[LoadedHead]:
    """Slice the shared evaluation table into one frame per rate head.

    ``predicted_points`` and ``actual_points`` there are the head's own
    target -- a per-90 rate, not points -- which is why they are renamed
    on the way out.
    """
    loaded = []
    for head in RATE_HEADS:
        run_id = runs.get(head.experiment)
        if run_id is None:
            logger.warning(
                "No scored run for %s; its component is left out.",
                head.experiment,
            )
            continue
        rows = points_rows.filter(pl.col("run_id") == run_id).select(
            *KEY_COLUMNS,
            "position",
            pl.col("predicted_points").alias("predicted_rate"),
            pl.col("actual_points").alias("actual_rate"),
        )
        if rows.is_empty():
            continue
        loaded.append(LoadedHead(head, rows))
    return loaded


def load_conceding_head(
    conceding_rows: pl.DataFrame, run_id: str | None, leg_teams: pl.DataFrame
) -> LoadedHead | None:
    """Fan the team-grain conceding rows out to the players who share them."""
    if run_id is None:
        logger.warning(
            "No scored run for %s; its component is left out.",
            CONCEDING_HEAD.experiment,
        )
        return None
    team_rows = conceding_rows.filter(pl.col("run_id") == run_id).select(
        "season",
        "gw",
        "team",
        "opposition",
        "predicted_conceded",
        "actual_conceded",
    )
    if team_rows.is_empty():
        return None
    rows = leg_teams.join(
        team_rows, on=["season", "gw", "team", "opposition"], how="inner"
    ).select(
        *KEY_COLUMNS,
        pl.col("predicted_conceded").alias("predicted_rate"),
        pl.col("actual_conceded").alias("actual_rate"),
    )
    return LoadedHead(CONCEDING_HEAD, rows)


def predicted_minutes(minutes_rows: pl.DataFrame, run_id: str) -> pl.DataFrame:
    """Return the minutes model's held-out forecast, keyed for a join."""
    return minutes_rows.filter(pl.col("run_id") == run_id).select(
        *KEY_COLUMNS,
        "expected_minutes",
        "p_zero",
        "p_partial",
        "p_sixty_plus",
    )


def oracle_minutes(actuals: pl.DataFrame) -> pl.DataFrame:
    """Return what a perfect minutes forecast would have said.

    The bucket probabilities collapse to a one-hot of what happened, so
    every component reading them -- appearance directly, the rate
    components through the exposure -- sees the truth.
    """
    played = pl.col("minutes") > 0
    hour = pl.col("minutes") >= 60
    return actuals.select(
        *KEY_COLUMNS,
        pl.col("minutes").cast(pl.Float64).alias("expected_minutes"),
        (~played).cast(pl.Float64).alias("p_zero"),
        (played & ~hour).cast(pl.Float64).alias("p_partial"),
        hour.cast(pl.Float64).alias("p_sixty_plus"),
    )


def _paid_positions(component: Component) -> list[str]:
    """Return the positions paid for a component.

    The conceding head fans out to every player of a club and the pooled
    rate heads serve several positions at once, so a head's rows reach
    positions that declare nothing for it -- a forward is paid neither
    for the sheet nor for the deduction. Composing those would be a
    component the position no longer declares, which compose() rejects.
    """
    return [
        position
        for position, declared in POSITION_COMPONENTS.items()
        if component in declared
    ]


def _component_frame(
    legs: pl.DataFrame, component: Component, points: pl.Expr
) -> pl.DataFrame:
    """Shape scored legs as ``points_component`` rows."""
    return legs.with_columns(
        prediction_kind=pl.lit(ABLATION_KIND),
        component=pl.lit(str(component)),
        points=points.cast(pl.Float64),
        model_version=pl.lit(str(component)),
        diagnostics=pl.lit(None, dtype=pl.Utf8),
    ).select(POINTS_COMPONENT.columns)


def _realised_count(rate: pl.Expr, minutes: pl.Expr) -> pl.Expr:
    """Return the count a per-90 rate was built from.

    The rate is the count scaled by the leg's own minutes, so undoing the
    scaling recovers the count exactly bar float error; rounding takes
    that off.
    """
    return (rate * minutes / 90.0).round()


def realised_points(component: Component, legs: pl.DataFrame) -> pl.Expr:
    """Return what a component actually paid, under FPL's rules.

    Not the component implementation with a perfect input: that would
    still read a perfect rate through a Poisson, so a head whose *shape*
    is wrong -- a threshold rule modelled as a survival probability, a
    floored count modelled as a mean -- would keep its error hidden
    inside the oracle. Foreknowledge means the points, not the rate.

    Parameters
    ----------
    component : Component
        Which component to price.
    legs : pl.DataFrame
        Legs carrying ``position``, ``minutes`` and ``actual_rate``
        (``actual_rate`` being goals conceded per match for conceding).

    Returns
    -------
    pl.Expr
        Realised points for the component, one per leg.

    Raises
    ------
    ValueError
        If the component has no realised pricing here.
    """
    minutes = pl.col("minutes").cast(pl.Float64)
    count = _realised_count(pl.col("actual_rate"), minutes)
    match component:
        case Component.APPEARANCE:
            return (
                pl.when(minutes >= 60)
                .then(2.0)
                .when(minutes > 0)
                .then(1.0)
                .otherwise(0.0)
            )
        case Component.GOALS:
            return count * pl.col("position").replace_strict(
                GOALS_POINTS_BY_POSITION, return_dtype=pl.Float64
            )
        case Component.ASSISTS:
            return count * ASSIST_POINTS
        case Component.YELLOW_CARDS:
            return count * YELLOW_CARD_POINTS
        case Component.SAVES:
            return (count // SAVES_DIVISOR).cast(pl.Float64)
        case Component.DEFCON:
            threshold = pl.col("position").replace_strict(
                DEFCON_THRESHOLD_BY_POSITION, return_dtype=pl.Float64
            )
            return (
                pl.when(count >= threshold).then(DEFCON_POINTS).otherwise(0.0)
            )
        case Component.CONCEDING:
            conceded = pl.col("actual_rate")
            # The team's total, not what was shipped while the player was
            # on: the component itself prices the clean sheet that way, so
            # the oracle has to as well or the difference would read as
            # model error.
            clean = (minutes >= 60) & (conceded == 0)
            price = pl.col("position").replace_strict(
                CLEAN_SHEET_POINTS_BY_POSITION, return_dtype=pl.Float64
            )
            docked = pl.col("position").is_in(CONCEDING_DEDUCTION_POSITIONS)
            deduction = (conceded // CONCEDING_DIVISOR).cast(pl.Float64)
            return pl.when(clean).then(price).otherwise(0.0) - pl.when(
                docked
            ).then(deduction).otherwise(0.0)
    raise ValueError(f"No realised pricing for component {component}.")


def _predicted_component(
    loaded: LoadedHead, legs: pl.DataFrame, minutes: pl.DataFrame
) -> pl.DataFrame:
    """Score a head's legs through the component implementation serving uses."""
    scored = legs.select(*KEY_COLUMNS, "position", "predicted_rate").rename(
        {"predicted_rate": PREDICTED_VALUE}
    )
    rows = loaded.head.impl.points(scored, minutes, ABLATION_KIND)
    return rows.with_columns(
        model_version=pl.lit(str(loaded.head.component))
    ).select(POINTS_COMPONENT.columns)


def _appearance_component(
    legs: pl.DataFrame, minutes: pl.DataFrame
) -> pl.DataFrame:
    """Score the appearance points, which are the minutes model's own output."""
    scored = legs.select(*KEY_COLUMNS, "position")
    rows = AppearanceComponent().points(scored, minutes, ABLATION_KIND)
    return rows.with_columns(
        model_version=pl.lit(str(Component.APPEARANCE))
    ).select(POINTS_COMPONENT.columns)


def build_universe(
    heads: Sequence[LoadedHead],
    actuals: pl.DataFrame,
    minutes: pl.DataFrame,
) -> pl.DataFrame:
    """Return the legs every variant can be scored on.

    A leg is in scope only when every component its position declares has
    a held-out prediction, a realised outcome, and a minutes forecast.
    The same legs then carry every variant, which is what makes the
    variants comparable at all -- a component whose oracle covered more
    rows than its prediction would score a difference that was really a
    change of population.
    """
    per_component: dict[Component, pl.DataFrame] = {}
    for loaded in heads:
        keys = loaded.rows.select(KEY_COLUMNS)
        existing = per_component.get(loaded.head.component)
        per_component[loaded.head.component] = (
            keys if existing is None else pl.concat([existing, keys]).unique()
        )
    legs = (
        actuals.select(KEY_COLUMNS)
        .join(minutes.select(KEY_COLUMNS), on=KEY_COLUMNS, how="inner")
        .join(
            _positions(heads),
            on=KEY_COLUMNS,
            how="inner",
        )
    )
    keep = []
    for position, declared in POSITION_COMPONENTS.items():
        rows = legs.filter(pl.col("position") == position)
        for component in declared:
            if component is Component.APPEARANCE:
                continue
            covered = per_component.get(component)
            if covered is None:
                rows = rows.clear()
                break
            rows = rows.join(covered, on=KEY_COLUMNS, how="semi")
        keep.append(rows)
    return pl.concat(keep) if keep else legs.clear()


def _positions(heads: Sequence[LoadedHead]) -> pl.DataFrame:
    """Return one position per leg, taken from whichever head carries it.

    The conceding head is team-grain and stamps no position of its own,
    so the player heads are the only source. They agree where they
    overlap: each reads the position off the same dummies.
    """
    frames = [
        loaded.rows.select(*KEY_COLUMNS, "position")
        for loaded in heads
        if "position" in loaded.rows.columns
    ]
    if not frames:
        return pl.DataFrame(
            schema={
                **{key: pl.Utf8 for key in KEY_COLUMNS},
                "position": pl.Utf8,
            }
        )
    return pl.concat(frames).unique(subset=KEY_COLUMNS)


def compose_variant(
    heads: Sequence[LoadedHead],
    universe: pl.DataFrame,
    actuals: pl.DataFrame,
    minutes: pl.DataFrame,
    oracle: Component | None,
) -> pl.DataFrame:
    """Compose one variant's totals over the shared leg universe.

    Parameters
    ----------
    heads : Sequence[LoadedHead]
        Every loaded head.
    universe : pl.DataFrame
        The legs to score, carrying ``position``.
    actuals : pl.DataFrame
        Realised minutes and points per leg.
    minutes : pl.DataFrame
        The minutes forecast the variant reads -- predicted for a
        component oracle, realised for the minutes variant.
    oracle : Component | None
        Which component to replace with what actually happened. None
        composes the baseline.

    Returns
    -------
    pl.DataFrame
        One composed prediction per leg, shaped like
        ``points_prediction``.
    """
    truth = universe.join(
        actuals.select(*KEY_COLUMNS, "minutes"), on=KEY_COLUMNS, how="inner"
    )
    rows: list[pl.DataFrame] = []
    if oracle is Component.APPEARANCE:
        rows.append(
            _component_frame(
                truth,
                Component.APPEARANCE,
                realised_points(Component.APPEARANCE, truth),
            )
        )
    else:
        rows.append(_appearance_component(universe, minutes))
    for loaded in heads:
        component = loaded.head.component
        legs = universe.filter(
            pl.col("position").is_in(_paid_positions(component))
        ).join(
            loaded.rows.drop("position", strict=False),
            on=KEY_COLUMNS,
            how="inner",
        )
        if legs.is_empty():
            continue
        if component is oracle:
            priced = legs.join(
                actuals.select(*KEY_COLUMNS, "minutes"),
                on=KEY_COLUMNS,
                how="inner",
            )
            rows.append(
                _component_frame(
                    priced, component, realised_points(component, priced)
                )
            )
        else:
            rows.append(_predicted_component(loaded, legs, minutes))
    composed = compose(
        pl.concat(rows, how="vertical"), expected=POSITION_COMPONENTS
    )
    return composed.join(
        universe.select(KEY_COLUMNS), on=KEY_COLUMNS, how="semi"
    )


def score(composed: pl.DataFrame, actuals: pl.DataFrame) -> pl.DataFrame:
    """Score composed totals per position and overall.

    Ranking metrics lead: the optimiser reads an order, not a level, so a
    component can move MAE without moving a single pick.
    """
    scored = composed.join(
        actuals.select(*KEY_COLUMNS, "actual_points"),
        on=KEY_COLUMNS,
        how="inner",
    ).select(
        "season",
        "gw",
        "position",
        pl.col("element").alias("player_id"),
        "predicted_points",
        pl.col("actual_points").alias("actual"),
    )
    rows = []
    for position in sorted(scored["position"].unique().to_list()):
        subset = scored.filter(pl.col("position") == position)
        rows.append(
            _metrics(subset, position, PRECISION_K_BY_POSITION[position])
        )
    if rows:
        # Overall precision@k is left out rather than faked: k is a
        # squad-slot count per position, and there is no k that means
        # anything across all four at once.
        rows.append(_metrics(scored, "ALL", None))
    return pl.DataFrame(rows)


def _metrics(
    subset: pl.DataFrame, position: str, k: int | None
) -> dict[str, object]:
    """Return one row of scores for one slice."""
    predicted = subset["predicted_points"].to_list()
    actual = subset["actual"].to_list()
    return {
        "position": position,
        "legs": subset.height,
        "mae": mae(predicted, actual) if actual else float("nan"),
        "rmse": rmse(predicted, actual) if actual else float("nan"),
        "spearman": spearman_by_gw(subset, gw_cols=RANKING_GROUP),
        "precision_at_k": (
            precision_at_k(subset, k=k, gw_cols=RANKING_GROUP)
            if k is not None
            else float("nan")
        ),
    }


def run_ablation(connection: "DuckDBPyConnection") -> pl.DataFrame:
    """Score the baseline and every oracle variant over one leg universe.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection holding the evaluation tables.

    Returns
    -------
    pl.DataFrame
        One row per variant and position, carrying the variant's scores
        and their deltas against the baseline.

    Raises
    ------
    ValueError
        If no leg carries a complete set of held-out components, which
        means the heads' folds do not overlap and there is nothing a
        composed total could be scored on.
    """
    points_rows = TEST_POINTS_PREDICTION.load(connection)
    conceding_rows = TEST_CONCEDING_PREDICTION.load(connection)
    minutes_rows = TEST_MINUTES_PREDICTION.load(connection)
    runs = latest_run_ids(
        [
            *points_rows["run_id"].unique().to_list(),
            *conceding_rows["run_id"].unique().to_list(),
            *minutes_rows["run_id"].unique().to_list(),
        ]
    )
    logger.info("Scoring the newest run per head: %s", runs)

    actuals = load_actuals(connection)
    heads = load_rate_heads(points_rows, runs)
    conceding = load_conceding_head(
        conceding_rows,
        runs.get(CONCEDING_HEAD.experiment),
        load_leg_teams(connection),
    )
    if conceding is not None:
        heads.append(conceding)

    minutes_run = runs.get("minutes_played_classification")
    if minutes_run is None:
        raise ValueError(
            "No scored minutes run. Every component reads the minutes "
            "forecast, so there is no baseline to ablate against."
        )
    forecast = predicted_minutes(minutes_rows, minutes_run)
    perfect = oracle_minutes(actuals)

    universe = build_universe(heads, actuals, forecast)
    if universe.is_empty():
        raise ValueError(
            "No fixture leg carries every component its position declares. "
            "The heads' folds do not overlap; retrain them over a common "
            "test window before ablating."
        )
    logger.info(
        "Scoring %d legs across %s.",
        universe.height,
        sorted(universe["position"].unique().to_list()),
    )

    baseline = score(
        compose_variant(heads, universe, actuals, forecast, None), actuals
    )
    results = [baseline.with_columns(variant=pl.lit(BASELINE))]
    components = sorted(
        {loaded.head.component for loaded in heads} | {Component.APPEARANCE},
        key=str,
    )
    for component in components:
        composed = compose_variant(
            heads, universe, actuals, forecast, component
        )
        results.append(
            score(composed, actuals).with_columns(
                variant=pl.lit(str(component))
            )
        )
    perfect_minutes = compose_variant(heads, universe, actuals, perfect, None)
    results.append(
        score(perfect_minutes, actuals).with_columns(
            variant=pl.lit(MINUTES_VARIANT)
        )
    )
    return _with_deltas(pl.concat(results))


def _with_deltas(results: pl.DataFrame) -> pl.DataFrame:
    """Attach each variant's movement against the baseline, per position."""
    baseline = results.filter(pl.col("variant") == BASELINE).select(
        "position",
        pl.col("mae").alias("_base_mae"),
        pl.col("spearman").alias("_base_spearman"),
        pl.col("precision_at_k").alias("_base_precision"),
    )
    return (
        results.join(baseline, on="position", how="left")
        .with_columns(
            delta_mae=pl.col("mae") - pl.col("_base_mae"),
            delta_spearman=pl.col("spearman") - pl.col("_base_spearman"),
            delta_precision_at_k=pl.col("precision_at_k")
            - pl.col("_base_precision"),
        )
        .drop("_base_mae", "_base_spearman", "_base_precision")
        .select(
            "variant",
            "position",
            "legs",
            "mae",
            "rmse",
            "spearman",
            "precision_at_k",
            "delta_mae",
            "delta_spearman",
            "delta_precision_at_k",
        )
        .sort("position", "delta_precision_at_k", descending=[False, True])
    )


def _declared_variants(position: str) -> set[str]:
    """Return the variants that can move a position's score.

    A component a position is not paid scores an unchanged total, and a
    row of exact zeros reads like measured indifference rather than a
    component that was never in the sum.
    """
    if position not in POSITION_COMPONENTS:
        return {str(component) for component in Component} | {
            BASELINE,
            MINUTES_VARIANT,
        }
    return {str(component) for component in POSITION_COMPONENTS[position]} | {
        BASELINE,
        MINUTES_VARIANT,
    }


def leverage_table(results: pl.DataFrame) -> pl.DataFrame:
    """Rank the variants by how much perfect knowledge of them would buy.

    Sorted on precision@k because that is what a squad is picked on. The
    baseline row is kept as the zero line rather than dropped.
    """
    allowed = pl.DataFrame(
        [
            {"position": position, "variant": variant}
            for position in results["position"].unique().to_list()
            for variant in sorted(_declared_variants(position))
        ]
    )
    return (
        results.join(allowed, on=["position", "variant"], how="semi")
        .sort(
            ["position", "delta_precision_at_k", "delta_spearman"],
            descending=[False, True, True],
        )
        .select(
            "position",
            "variant",
            "legs",
            pl.col("delta_precision_at_k").round(4),
            pl.col("delta_spearman").round(4),
            pl.col("delta_mae").round(4),
        )
    )


def summarise(results: pl.DataFrame) -> str:
    """Return the leverage table as text, one block per position."""
    blocks = []
    for position in results["position"].unique(maintain_order=True).to_list():
        subset = leverage_table(results).filter(pl.col("position") == position)
        blocks.append(f"{position}\n{subset.drop('position')}")
    return "\n\n".join(blocks)
