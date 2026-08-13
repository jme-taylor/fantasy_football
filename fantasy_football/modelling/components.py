"""The scoring components a points prediction is summed from.

A prediction used to be one number from one model. It is now a sum of
components -- ``E[total] = sum of E[component]``, which holds by
linearity even though the components correlate. A position that has not
been decomposed is the degenerate case of a single ``TOTAL`` component,
so nothing downstream needs to ask whether a position has been split up.

Components live as rows in ``points_component`` rather than as a table
each, so adding one is a new :class:`Component` value and an entry in
:data:`POSITION_COMPONENTS`.
"""

import logging
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING

import polars as pl

from fantasy_football.storage.tables import POINTS_COMPONENT, POINTS_PREDICTION

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)


class Component(StrEnum):
    """One scoring component of an FPL points total."""

    #: A whole undecomposed prediction, for positions not yet split up.
    TOTAL = "total"
    #: Points for playing at all, and for playing an hour.
    APPEARANCE = "appearance"
    #: Points for clearing the defensive-contribution threshold.
    DEFCON = "defcon"
    #: Everything not carved out into a component of its own.
    RESIDUAL = "residual"


# Which components each position's prediction is summed from. Read as
# configuration rather than branched on in code, so putting a position
# back to a single model is an edit here and not a revert.
POSITION_COMPONENTS: dict[str, tuple[Component, ...]] = {
    "GK": (Component.TOTAL,),
    "DEF": (Component.TOTAL,),
    "MID": (Component.TOTAL,),
    "FWD": (Component.TOTAL,),
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


def _reject_incomplete_predictions(
    rows: pl.DataFrame, expected: Mapping[str, Sequence[Component]]
) -> None:
    """Raise if a fixture leg is missing a component its position needs.

    Half a decomposition sums to a quietly low prediction, which the
    optimiser acts on without complaint. This is the check that turns
    that into a failure.
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
    for row in present.iter_rows(named=True):
        wanted = {str(component) for component in expected[row["position"]]}
        missing = sorted(wanted - set(row["components"]))
        if missing:
            raise ValueError(
                f"Prediction for {row['element']} in {row['season']} gw "
                f"{row['gw']} is missing components {missing}, so it would "
                "sum to less than the model predicts."
            )


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
        Which components each position must supply. When given, a
        prediction missing one of them is an error. Defaults to None,
        meaning completeness is not checked.

    Returns
    -------
    pl.DataFrame
        Rows shaped like ``points_prediction``.

    Raises
    ------
    ValueError
        If a component is unknown, carried twice for one fixture leg, or
        missing from a position that declares it.
    """
    if component_rows.is_empty():
        return POINTS_PREDICTION.coerce(
            pl.DataFrame(schema=POINTS_PREDICTION.schema)
        )
    rows = POINTS_COMPONENT.coerce(component_rows)
    _reject_unknown_components(rows)
    _reject_duplicate_components(rows)
    if expected is not None:
        _reject_incomplete_predictions(rows, expected)
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
