"""Shared machinery for the per-position points regressors.

Every position predicts the same target (``total_points``) from the same
shape of features: the player's own rolling form, his club's rolling
form, the opposition's rolling form, and the minutes forecast. What
differs between positions is only *which* columns are used -- a defender
cares about clean sheets and xG conceded, a forward about xG for -- and
which rows are in scope.

Subclasses therefore declare column lists and nothing else. Everything
that reads those lists lives here, so a fix to the as-of join or the
partition scoping lands on every position at once rather than being
copied four times.
"""

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar, override

import numpy as np
import polars as pl
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fantasy_football.constants import PRECISION_K_BY_POSITION
from fantasy_football.features.match_form import rolling_identity_sql
from fantasy_football.features.views import register_feature_views
from fantasy_football.modelling.components import (
    PREDICTED_VALUE,
    Component,
    PointsComponent,
)
from fantasy_football.modelling.folds import Fold, FoldResult
from fantasy_football.modelling.metrics import (
    Metrics,
    PointsMetrics,
    mae,
    precision_at_k,
    rmse,
    skill_score,
    spearman_by_gw,
)
from fantasy_football.modelling.predictor import Predictor, feature_json
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    MINUTES_PREDICTION,
    POINTS_COMPONENT,
    TEAM_FIXTURE,
)

if TYPE_CHECKING:
    from duckdb import DuckDBPyConnection

logger = logging.getLogger(__name__)

# The whole-points target every position predicted before the models
# were decomposed into components. Subclasses override
# ``PositionPointsPredictor.TARGET`` when they predict something narrower.
TARGET = "total_points"

# Match-grain keys. They identify a row and are never model inputs.
KEY_COLUMNS = ["season", "gw", "element", "opponent"]

# What counts as one gameweek for the ranking metrics.
RANKING_GROUP = ("season", "gw")

# The player-form columns that reset each season. Everything else a
# position lists in ``PLAYER_FORM_COLUMNS`` is a rolling rate that
# deliberately spans seasons.
SEASON_TO_DATE_COLUMNS = [
    "yellow_cards_season_to_date",
    "red_cards_season_to_date",
]

# How stale the carried form is. Never as-of joined like the rates: the
# inclusive view reports it against the appearance's own kickoff, which
# is always 0, so the forward path measures it against the fixture's
# kickoff instead.
STALENESS_COLUMN = "days_since_last_appearance"

# Holds the matched appearance's kickoff long enough to measure that gap.
_LAST_APPEARANCE = "last_appearance_kickoff"


def asof_form(
    left: pl.DataFrame,
    right: pl.DataFrame,
    by: list[str],
    columns: list[str],
) -> pl.DataFrame:
    """Attach the most recent form row at or before each fixture.

    Parameters
    ----------
    left : pl.DataFrame
        Forward fixtures, carrying ``kickoff_time`` and every column in
        ``by``.
    right : pl.DataFrame
        A form frame carrying ``kickoff_time``, ``by`` and ``columns``.
    by : list[str]
        Columns matched exactly before the as-of comparison. They must
        identify one subject -- a player or a club -- or a fixture would
        inherit a stranger's form.
    columns : list[str]
        Form columns to bring across.

    Returns
    -------
    pl.DataFrame
        ``left``, ordered by ``kickoff_time``, with the form columns
        attached. Exactly one row per input row: an as-of join takes at
        most one match, so a duplicated right-hand row cannot fan the
        output out -- it only changes which value arrives.
    """
    # Typed from the right-hand frame where possible: a carried kickoff
    # is a datetime, and defaulting it to Float64 would break the
    # staleness arithmetic on an empty frame.
    absent = [
        pl.lit(None, dtype=right.schema.get(name, pl.Float64)).alias(name)
        for name in columns
    ]
    if right.is_empty():
        return left.with_columns(absent)
    # Drop rows with no kickoff time
    slimmed = (
        right.select(by + ["kickoff_time"] + columns)
        .with_columns(pl.col("kickoff_time").cast(pl.Datetime("us")))
        .filter(pl.col("kickoff_time").is_not_null())
        .sort("kickoff_time")
    )
    if slimmed.is_empty():
        return left.with_columns(absent)
    return (
        left.with_columns(pl.col("kickoff_time").cast(pl.Datetime("us")))
        .sort("kickoff_time")
        .join_asof(
            slimmed,
            on="kickoff_time",
            by=by,
            strategy="backward",
            check_sortedness=False,
        )
    )


def attach_rolling_identity(
    connection: "DuckDBPyConnection", frame: pl.DataFrame
) -> pl.DataFrame:
    """Add the form views' ``rolling_identity`` to forward fixture rows.

    Resolved through ``player_season`` by the same expression the view
    uses, rather than reimplemented here, so serve-time identity cannot
    drift from the identity the rolling window partitions by. A player
    with no ``player_season`` row falls back to the view's own
    ``(season, element)`` form, which the LEFT JOIN reaches because the
    fallback branch reads its season and element from the fixture side.

    Parameters
    ----------
    connection : duckdb.DuckDBPyConnection
        An open connection holding ``player_season``.
    frame : pl.DataFrame
        Forward fixture rows carrying ``season`` and ``element``.

    Returns
    -------
    pl.DataFrame
        ``frame`` with a non-null ``rolling_identity`` string column.
        ``player_season`` is keyed on ``(season, element)``, so the join
        cannot add rows.
    """
    keys = frame.select("season", "element").unique()
    connection.register("_forward_identity_keys", keys)
    identities = connection.sql(
        "SELECT k.season, k.element, "
        f"{rolling_identity_sql('ps', 'k')} AS rolling_identity "
        "FROM _forward_identity_keys AS k "
        "LEFT JOIN player_season AS ps "
        "ON ps.season = k.season AND ps.element = k.element"
    ).pl()
    connection.unregister("_forward_identity_keys")
    return frame.join(identities, on=["season", "element"], how="left")


#: The indicator column standing for each position, for models pooled
#: across more than one. A tree cannot split on a string, and the shared
#: pipeline carries no categorical encoder, so these are selected as
#: dummies in SQL the same way ``is_home`` is.
POSITION_DUMMIES: dict[str, str] = {
    "GK": "is_goalkeeper",
    "DEF": "is_defender",
    "MID": "is_midfielder",
    "FWD": "is_forward",
}


def position_dummy_names(positions: Sequence[str]) -> list[str]:
    """Return the indicator column names for a pooled model's positions.

    Empty for a single position: the column would be constant, which
    tells a model nothing and costs a split to discover.

    Parameters
    ----------
    positions : Sequence[str]
        The positions a model trains on.

    Returns
    -------
    list[str]
        One name per position, in the order given, or empty.
    """
    if len(positions) < 2:
        return []
    return [POSITION_DUMMIES[position] for position in positions]


class PositionPointsPredictor(Predictor):
    """Points regressor for one FPL position.

    Subclasses set the class variables below and inherit everything
    else. ``POSITION`` must match the ``position`` on the subclass's
    :class:`~fantasy_football.modelling.predictor.ModelSpec`, since that
    is what keeps each position's rows in ``points_prediction`` from
    overwriting another's.

    ``POSITION`` is the position this model *serves*, which is not
    necessarily the population it learns from: see
    ``TRAINING_POSITIONS``. Neither ``POSITION`` nor the spec's
    ``position`` describes the training population.
    """

    #: The ``player_season.position`` value this model serves. Scoped
    #: scoring, writing and evaluation, not training.
    POSITION: ClassVar[str]
    #: The positions this model trains on, or empty for ``POSITION``
    #: alone. A pooled model learns from positions it never writes a
    #: prediction for -- goals are the same event whoever scores them,
    #: and a defender's are far too rare to estimate on their own.
    TRAINING_POSITIONS: ClassVar[tuple[str, ...]] = ()
    #: The positions this model writes predictions for, or empty for
    #: ``POSITION`` alone. Must be a subset of ``TRAINING_POSITIONS``: a
    #: model cannot serve a position it never saw, and the dummy telling
    #: the positions apart only exists when it trained on more than one.
    #: ``POSITION`` stays the model's primary one, which is what keys its
    #: registered name and its evaluation.
    SERVING_POSITIONS: ClassVar[tuple[str, ...]] = ()
    #: The column this model predicts.
    TARGET: ClassVar[str] = TARGET
    #: The scoring component this model's predictions are stored as.
    #: Required: a default would let a head write its rows under
    #: another component's name, which composes into a wrong total
    #: rather than failing.
    COMPONENT: ClassVar[Component]
    #: Model inputs, in the order the frame presents them.
    FEATURES: ClassVar[list[str]]
    #: Per-player form columns read from the player form views.
    PLAYER_FORM_COLUMNS: ClassVar[list[str]]
    #: Form columns describing the player's own club.
    OWN_TEAM_COLUMNS: ClassVar[list[str]]
    #: Form columns describing the club he is playing against.
    OPPOSITION_COLUMNS: ClassVar[list[str]]
    #: Columns taken from the minutes model's predictions.
    MINUTES_COLUMNS: ClassVar[list[str]]
    #: Seasons this position may train on, or None for no restriction.
    TRAINING_SEASONS: ClassVar[tuple[str, ...] | None] = None
    #: Prefix applied to the opposition copies of the team form columns.
    OPPOSITION_PREFIX: ClassVar[str] = ""
    #: Turns this model's output into stored component points.
    COMPONENT_IMPL: ClassVar[PointsComponent]
    #: Column weighting each training row, or None to weight equally.
    #: Per-90 targets need this: two CBIT in eight minutes is a rate of
    #: 22.5, and unweighted those rows dominate the fit.
    WEIGHT_COLUMN: ClassVar[str | None] = None
    #: Extra ``expression AS name`` selections the frame carries beyond
    #: keys, target and features -- scoring inputs rather than model
    #: inputs, so they must never appear in ``FEATURES``.
    EXTRA_COLUMNS: ClassVar[tuple[str, ...]] = ()
    #: Values to fill a feature's nulls with, by feature name. Only for
    #: features whose null has a known meaning; everything else is left
    #: to the pipeline's imputer, which learns the fill from the data.
    FEATURE_FILLS: ClassVar[dict[str, float]] = {}

    @property
    def training_positions(self) -> tuple[str, ...]:
        """Return the positions this model's training frame spans."""
        return self.TRAINING_POSITIONS or (self.POSITION,)

    @property
    @override
    def serving_positions(self) -> tuple[str, ...]:
        """Return the positions this model writes predictions for.

        Raises
        ------
        ValueError
            If a served position was never trained on. The dummies that
            tell positions apart are built from the training list, so
            serving outside it would stamp every row with whichever
            position matched nothing.
        """
        served = self.SERVING_POSITIONS or (self.POSITION,)
        unseen = sorted(set(served) - set(self.training_positions))
        if unseen:
            raise ValueError(
                f"{type(self).__name__} serves {unseen} without training "
                f"on them; TRAINING_POSITIONS is {self.training_positions}."
            )
        return served

    def _served_position(self) -> pl.Expr:
        """Return each row's own position, read off its dummies.

        A model serving one position stamps a literal, as it always did.
        One serving several must stamp what the row actually is, or the
        rows land under the wrong position and ``compose`` cannot tell:
        every component name involved is legitimate.
        """
        served = self.serving_positions
        if len(served) < 2:
            return pl.lit(self.POSITION)
        expression = pl.lit(None, dtype=pl.Utf8)
        for position in served:
            expression = (
                pl.when(pl.col(POSITION_DUMMIES[position]) == 1.0)
                .then(pl.lit(position))
                .otherwise(expression)
            )
        return expression

    def fill_features(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Apply ``FEATURE_FILLS`` to whichever columns are present."""
        return frame.with_columns(
            [
                pl.col(name).fill_null(value)
                for name, value in self.FEATURE_FILLS.items()
                if name in frame.columns
            ]
        )

    @property
    @override
    def expected_features(self) -> list[str] | None:
        """Return this position's declared model inputs."""
        return self.FEATURES

    @property
    def target_sql(self) -> str:
        """Return the SELECT expression producing the target column."""
        return f"m.{self.TARGET}"

    @property
    def extra_joins(self) -> str:
        """Return any joins beyond the shared ones. Empty by default."""
        return ""

    @property
    def row_filter(self) -> str:
        """Return an extra WHERE predicate. Empty by default.

        Applied when fitting *and* when scoring, so it governs which
        fixture legs this component exists for at all. A leg missing one
        of its position's components is dropped from the composed
        prediction entirely, so narrowing this narrows the whole
        position's output -- put fit-only restrictions in
        ``training_row_filter`` instead.
        """
        return ""

    @property
    def training_row_filter(self) -> str:
        """Return a WHERE predicate applied when fitting only.

        For rows that would teach the model something false but still
        have to be scored: a cameo whose per-90 rate is arithmetic
        noise, or a debut with no form behind it.
        """
        return ""

    @property
    def frame_columns(self) -> list[str]:
        """Return every non-feature column the training frame carries."""
        extra = [
            selection.split(" AS ")[-1] for selection in self.EXTRA_COLUMNS
        ]
        if self.WEIGHT_COLUMN and self.WEIGHT_COLUMN not in extra:
            extra.append(self.WEIGHT_COLUMN)
        return extra

    def sample_weight(self, frame: pl.DataFrame) -> list[float] | None:
        """Return per-row fitting weights, or None to weight equally."""
        if self.WEIGHT_COLUMN is None:
            return None
        return (
            frame[self.WEIGHT_COLUMN].cast(pl.Float64).fill_null(0.0).to_list()
        )

    def _fit_weights(self, frame: pl.DataFrame) -> dict[str, list[float]]:
        """Return the ``fit`` keyword arguments carrying sample weights.

        Empty when the model weights rows equally, so the unweighted
        positions call ``fit`` exactly as they did before weighting
        existed.
        """
        weights = self.sample_weight(frame)
        if weights is None:
            return {}
        return {"model__sample_weight": weights}

    def minutes_predictions(self, kind: str) -> pl.DataFrame:
        """Return the minutes forecast this prediction kind should read.

        Minutes reach a component here and nowhere else. No component
        model may carry a minutes feature, so this is the single point at
        which the minutes forecast enters a points number.
        """
        return MINUTES_PREDICTION.load(self.connection).filter(
            pl.col("prediction_kind") == kind
        )

    @property
    def opposition_feature_names(self) -> list[str]:
        """Return the opposition form columns as the model sees them."""
        return [
            f"{self.OPPOSITION_PREFIX}{column}"
            for column in self.OPPOSITION_COLUMNS
        ]

    @property
    def season_to_date_columns(self) -> list[str]:
        """Return the player-form columns that reset each season."""
        return [
            column
            for column in self.PLAYER_FORM_COLUMNS
            if column in SEASON_TO_DATE_COLUMNS
        ]

    @property
    def player_rolling_columns(self) -> list[str]:
        """Return the player-form columns that span seasons."""
        return [
            column
            for column in self.PLAYER_FORM_COLUMNS
            if column not in SEASON_TO_DATE_COLUMNS
        ]

    def model_frame_sql(self, training: bool = True) -> str:
        """Return the SELECT behind this position's model frame.

        Joins the match-grain target on ``player_match`` to the minutes
        forecast, the player's rolling form, and both teams' rolling
        form. ``player_week`` supplies the player's club for the season,
        which is what resolves which side of ``team_match_form`` is his
        own.

        ``TRAINING_SEASONS``, where a position sets it, adds a season
        predicate. Positions leaving it None generate exactly the SQL
        they generated before it existed.

        Parameters
        ----------
        training : bool, optional
            When False, ``training_row_filter`` is left off, so the
            frame spans every leg this component must be scored for
            rather than only the ones it may learn from.

        Returns
        -------
        str
            A SELECT over ``player_match`` and the registered feature
            views.
        """
        # A decomposed model declares none of these: minutes reach it at
        # composition, never as a feature. The join stays either way, so
        # the SQL differs only by the selected columns.
        minutes = "".join(
            f"\n    mn.{column}," for column in self.MINUTES_COLUMNS
        )
        own = ",\n    ".join(
            f"own.{column} AS {column}" for column in self.OWN_TEAM_COLUMNS
        )
        opposition = ",\n    ".join(
            f"opp.{column} AS {alias}"
            for column, alias in zip(
                self.OPPOSITION_COLUMNS,
                self.opposition_feature_names,
                strict=True,
            )
        )
        player = ",\n    ".join(
            f"mf.{column} AS {column}" for column in self.PLAYER_FORM_COLUMNS
        )
        # Joined rather than interpolated one per line: a head that reads
        # no own-team column is legitimate, and a fixed three-line layout
        # would leave a dangling comma behind the empty one.
        form = ",\n    ".join(
            part for part in (player, own, opposition) if part
        )
        extra = "".join(
            f"\n    {selection}," for selection in self.EXTRA_COLUMNS
        )
        seasons = ""
        if self.TRAINING_SEASONS is not None:
            # An empty tuple is the reachable failure
            if not self.TRAINING_SEASONS:
                raise ValueError(
                    f"{type(self).__name__} sets TRAINING_SEASONS to an "
                    "empty tuple, so it would train on no rows at all. "
                    "This usually means the stat lists it derives the "
                    "window from have no season in common -- check their "
                    "coverage in storage/coverage.py. Pass None to train "
                    "on every season."
                )
            named = ", ".join(
                f"'{season}'" for season in self.TRAINING_SEASONS
            )
            seasons = f"\n  AND m.season IN ({named})"
        fit_only = self.training_row_filter if training else ""
        positions = ", ".join(
            f"'{position}'" for position in self.training_positions
        )
        dummies = "".join(
            f"\n    CAST(s.position = '{position}' AS DOUBLE) AS {name},"
            for position, name in zip(
                self.training_positions,
                position_dummy_names(self.training_positions),
                strict=False,
            )
        )
        return f"""
SELECT
    m.season,
    m.gw,
    m.element,
    m.opponent,
    {self.target_sql},{extra}{dummies}
    m.is_home,{minutes}
    {form}
FROM player_match AS m
INNER JOIN player_season AS s
    ON  m.element = s.element
    AND m.season  = s.season
    AND s.position IN ({positions})
-- prediction_kind is part of the minutes primary key, so a fixture that
-- was forward-scored before it was played and backfilled afterwards
-- carries both kinds. Joining unfiltered would fan the training row out.
-- Backfill is the right kind here: see the train/serve skew TODO in the
-- subclass module.
LEFT JOIN minutes_prediction AS mn
    ON  m.season   = mn.season
    AND m.gw       = mn.gw
    AND m.element  = mn.element
    AND m.opponent = mn.opponent
    AND mn.prediction_kind = '{BACKFILL_KIND}'
LEFT JOIN player_match_form AS mf
    ON  m.season   = mf.season
    AND m.gw       = mf.gw
    AND m.element  = mf.element
    AND m.opponent = mf.opponent
LEFT JOIN player_week AS pw
    ON  pw.season  = m.season
    AND pw.gw      = m.gw
    AND pw.element = m.element
LEFT JOIN fpl_team_id AS opp_id
    ON  opp_id.season  = m.season
    AND opp_id.team_id = m.opponent
LEFT JOIN team_match_form AS own
    ON  own.season     = m.season
    AND own.gw         = m.gw
    AND own.team       = pw.team
    AND own.opposition = opp_id.team
LEFT JOIN team_match_form AS opp
    ON  opp.season     = m.season
    AND opp.gw         = m.gw
    AND opp.team       = opp_id.team
    AND opp.opposition = pw.team{self.extra_joins}
WHERE m.minutes IS NOT NULL{seasons}{self.row_filter}{fit_only}
"""

    @override
    def build_training_data(self) -> pl.DataFrame:
        """Build this position's training frame from the store.

        Registers the feature views first, so callers do not have to
        remember to.

        Returns
        -------
        pl.DataFrame
            One row per played fixture leg at this position, with
            ``KEY_COLUMNS``, ``TARGET`` and ``FEATURES``, in that order.
        """
        register_feature_views(self.connection)
        frame = self.fill_features(
            self.connection.sql(self.model_frame_sql()).pl()
        )
        return frame.select(
            KEY_COLUMNS + [self.TARGET] + self.frame_columns + self.FEATURES
        )

    @override
    def scoring_frame(self) -> pl.DataFrame:
        """Return every leg this model must write a component for.

        Two things separate this from the training frame, and both are
        silent if they are missed. It keeps the legs
        ``training_row_filter`` drops, because a leg missing one of its
        position's components is dropped from the composed prediction
        altogether -- a stricter fit would otherwise delete predictions
        the other components were happy to make. And it restricts a
        pooled model to the position it serves, because the written rows
        are stamped ``POSITION`` regardless of which position they came
        from, so a pooled model would file every midfielder as a
        defender and price his goals at six points.
        """
        if not self._scoring_differs_from_training:
            return super().scoring_frame()
        if self._scoring_dataframe is None:
            register_feature_views(self.connection)
            frame = self.fill_features(
                self.connection.sql(self.model_frame_sql(training=False)).pl()
            ).select(
                KEY_COLUMNS
                + [self.TARGET]
                + self.frame_columns
                + self.FEATURES
            )
            self._scoring_dataframe = self._own_position_rows(frame)
        return self._scoring_dataframe

    @property
    def _scoring_differs_from_training(self) -> bool:
        """Whether this model scores rows it does not train on.

        Only two things make it so, and a model with neither reuses the
        training frame exactly as it did before the split existed.
        """
        return (
            bool(self.training_row_filter) or len(self.training_positions) > 1
        )

    def _own_position_rows(self, frame: pl.DataFrame) -> pl.DataFrame:
        """Keep only the rows belonging to the positions this serves.

        Read off the position indicators rather than a position column,
        which the frame does not carry. A model spanning one position
        has no indicator and needs no filter -- every row is already
        its own.
        """
        names = position_dummy_names(self.training_positions)
        if not names:
            return frame
        return frame.filter(
            pl.any_horizontal(
                [
                    pl.col(POSITION_DUMMIES[position]) == 1.0
                    for position in self.serving_positions
                ]
            )
        )

    def make_pipeline(self) -> Pipeline:
        """Build the median-imputing random-forest pipeline.

        Imputation lives inside the pipeline so it refits per fold on
        training data only. There is deliberately no scaler: it is a
        no-op for trees, and the notebook only carried one because it
        started with a linear model.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted pipeline ending in a ``RandomForestRegressor``.
        """
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", RandomForestRegressor(n_jobs=-1)),
            ]
        )

    def fold_metrics(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> Metrics:
        """Score one fold's predictions against its actuals.

        The baseline for the skill score is the mean target over the
        fold, i.e. "predict the average player at this position every
        time". Ranking metrics matter more than MAE here: the optimiser
        only needs the order to be right.

        Parameters
        ----------
        test_df : pl.DataFrame
            The fold's held-out rows, carrying ``KEY_COLUMNS`` and
            ``TARGET``.
        predicted : Sequence[float]
            Predictions aligned to ``test_df`` row order.

        Returns
        -------
        Metrics
            The fold's scores. Positions predicting a component rather
            than whole points return their own container.
        """
        actual = test_df[self.TARGET].to_list()
        baseline = [float(np.mean(actual))] * len(actual)
        # Season joins gw as the ranking group because a holdout spans
        # seasons, and gameweek numbers repeat each year.
        ranked = test_df.select(
            pl.col("season"),
            pl.col("gw"),
            pl.col("element").alias("player_id"),
            pl.Series("predicted_points", predicted),
            pl.col(self.TARGET).alias("actual"),
        )
        return PointsMetrics(
            mae=mae(predicted, actual),
            rmse=rmse(predicted, actual),
            skill_score=skill_score(predicted, actual, baseline),
            spearman=spearman_by_gw(ranked, gw_cols=RANKING_GROUP),
            # TODO (JT): Make k a few different values
            precision_at_k=precision_at_k(
                ranked,
                k=PRECISION_K_BY_POSITION[self.POSITION],
                gw_cols=RANKING_GROUP,
            ),
        )

    @override
    def build_forward_data(
        self, forward_fixtures: pl.DataFrame
    ) -> pl.DataFrame:
        """Build features for this position's unplayed fixtures.

        Parameters
        ----------
        forward_fixtures : pl.DataFrame
            Rows from
            :func:`fantasy_football.modelling.forward.build_forward_fixtures`,
            covering every position.

        Returns
        -------
        pl.DataFrame
            One row per in-scope fixture, with ``KEY_COLUMNS`` and
            ``FEATURES``.
        """
        # The training frame restricts to this position in SQL, so the
        # forward path must too. Without it every rostered player is
        # scored and stamped POSITION, and the other positions' models
        # collide with these rows in points_prediction.
        forward_fixtures = forward_fixtures.filter(
            pl.col("position").is_in(self.serving_positions)
        )
        player_form = self.connection.sql(
            "SELECT * FROM player_match_form_inclusive"
        ).pl()
        team_form = self.connection.sql(
            "SELECT * FROM team_match_form_inclusive"
        ).pl()
        fixtures = TEAM_FIXTURE.load(self.connection).select(
            "season",
            "gw",
            "team",
            "kickoff_time",
            "is_home",
            pl.col("opposition").alias("opponent_name"),
        )
        minutes = MINUTES_PREDICTION.load(self.connection).filter(
            pl.col("prediction_kind") == FORWARD_KIND
        )
        frame = forward_fixtures.join(
            fixtures,
            on=["season", "gw", "team", "kickoff_time"],
            how="left",
        ).join(
            minutes.select(KEY_COLUMNS + self.MINUTES_COLUMNS),
            on=KEY_COLUMNS,
            how="left",
        )

        frame = attach_rolling_identity(self.connection, frame)
        # Rolling rates carry across the season boundary, so they match
        # on identity alone. The season-to-date counts must not, so they
        # match on identity *and* season and coalesce to zero -- a player
        # whose last appearance was last season starts this one on nil,
        # not on last season's closing tally.
        rolling = [
            column
            for column in self.player_rolling_columns
            if column != STALENESS_COLUMN
        ]
        frame = asof_form(
            frame,
            player_form.with_columns(
                pl.col("kickoff_time").alias(_LAST_APPEARANCE)
            ),
            ["rolling_identity"],
            [*rolling, _LAST_APPEARANCE],
        ).with_columns(
            (pl.col("kickoff_time") - pl.col(_LAST_APPEARANCE))
            .dt.total_days()
            .cast(pl.Float64)
            .alias(STALENESS_COLUMN)
        )
        frame = frame.drop(_LAST_APPEARANCE)
        frame = asof_form(
            frame,
            player_form,
            ["rolling_identity", "season"],
            self.season_to_date_columns,
        ).with_columns(
            [
                pl.col(column).fill_null(0.0)
                for column in self.season_to_date_columns
            ]
        )

        frame = asof_form(frame, team_form, ["team"], self.OWN_TEAM_COLUMNS)
        opposition = team_form.select(
            pl.col("team").alias("opponent_name"),
            "kickoff_time",
            *[
                pl.col(column).alias(alias)
                for column, alias in zip(
                    self.OPPOSITION_COLUMNS,
                    self.opposition_feature_names,
                    strict=True,
                )
            ],
        )
        frame = asof_form(
            frame, opposition, ["opponent_name"], self.opposition_feature_names
        )
        frame = frame.with_columns(pl.col("is_home").cast(pl.Float64))
        # Read off the row's own position rather than stamped from
        # POSITION: the forward frame was filtered to the served
        # positions a few lines up, which may be more than one.
        frame = frame.with_columns(
            [
                (pl.col("position") == position).cast(pl.Float64).alias(name)
                for position, name in zip(
                    self.training_positions,
                    position_dummy_names(self.training_positions),
                    strict=False,
                )
            ]
        )
        frame = self.fill_features(frame)
        missing = [name for name in self.FEATURES if name not in frame.columns]
        if missing:
            frame = frame.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias(name)
                    for name in missing
                ]
            )
            logger.warning(
                "Forward %s features absent from the join: %s",
                self.POSITION,
                missing,
            )
        return frame.select(KEY_COLUMNS + self.FEATURES)

    def fold_predictions(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> pl.DataFrame:
        """Shape one fold's scored rows for the evaluation table.

        ``run_id`` is left off: the fold does not know which run it
        belongs to, and the predictor stamps it on the way to storage.

        Parameters
        ----------
        test_df : pl.DataFrame
            The fold's held-out rows.
        predicted : Sequence[float]
            Predictions aligned to ``test_df`` row order.

        Returns
        -------
        pl.DataFrame
            Keys, position, prediction, actual and the model's inputs.
        """
        return test_df.with_columns(
            position=self._served_position(),
            predicted_points=pl.Series(predicted).cast(pl.Float64),
            actual_points=pl.col(self.TARGET).cast(pl.Float64),
            features=feature_json(test_df, self.FEATURES),
        ).select(
            KEY_COLUMNS
            + ["position", "predicted_points", "actual_points", "features"]
        )

    @override
    def fit_predict_fold(self, fold: Fold) -> FoldResult:
        """Fit on the fold's train split and score its test split."""
        pipe = self.make_pipeline()
        pipe.fit(
            fold.train.select(self.FEATURES).to_pandas(),
            fold.train[self.TARGET].to_list(),
            **self._fit_weights(fold.train),
        )
        predicted = list(
            pipe.predict(fold.test.select(self.FEATURES).to_pandas())
        )
        return FoldResult(
            metrics=self.fold_metrics(fold.test, predicted),
            predictions=self.fold_predictions(fold.test, predicted),
        )

    @override
    def train_final(self, feature_frame: pl.DataFrame) -> Pipeline:
        """Fit the pipeline on every row in ``feature_frame``."""
        pipe = self.make_pipeline()
        pipe.fit(
            feature_frame.select(self.FEATURES).to_pandas(),
            feature_frame[self.TARGET].to_list(),
            **self._fit_weights(feature_frame),
        )
        return pipe

    @override
    def build_prediction_rows(
        self,
        feature_frame: pl.DataFrame,
        model: Pipeline,
        version: str,
        kind: str,
    ) -> pl.DataFrame:
        """Score ``feature_frame`` and shape it as this model's component.

        The rows are one scoring component, not a whole prediction. What
        the optimiser reads is the sum of every component a position
        declares, which
        :func:`fantasy_football.modelling.components.compose` builds.
        """
        predicted = model.predict(
            feature_frame.select(self.FEATURES).to_pandas()
        )
        # Positioned before the select, not after: the dummies the
        # served position is read off are model features, and selecting
        # the keys first would drop them.
        scored = feature_frame.with_columns(
            position=self._served_position(),
            **{PREDICTED_VALUE: pl.Series(predicted).cast(pl.Float64)},
        ).select([*KEY_COLUMNS, "position", PREDICTED_VALUE])
        rows = self.COMPONENT_IMPL.points(
            scored, self.minutes_predictions(kind), kind
        )
        return rows.with_columns(model_version=pl.lit(version)).select(
            POINTS_COMPONENT.columns
        )
