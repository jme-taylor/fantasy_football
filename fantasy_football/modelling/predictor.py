import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import duckdb
import mlflow
import mlflow.sklearn
import polars as pl
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict
from sklearn.pipeline import Pipeline

from fantasy_football.constants import CURRENT_SEASON, MLFLOW_TRACKING_URI
from fantasy_football.extraction.fpl import FplAPI
from fantasy_football.extraction.snapshot import warn_unidentified_snapshot
from fantasy_football.features.roster import latest_snapshot
from fantasy_football.modelling.folds import Fold, FoldResult, FoldStrategy
from fantasy_football.modelling.forward import (
    build_forward_fixtures,
    last_played_gw,
)
from fantasy_football.modelling.metrics import summarise
from fantasy_football.modelling.registry import load_production_model
from fantasy_football.storage.tables import (
    BACKFILL_KIND,
    FORWARD_KIND,
    PLAYER_SEASON,
    PLAYER_SNAPSHOT,
    PLAYER_WEEK,
    TEAM_FIXTURE,
    Table,
    prediction_versions,
)

load_dotenv()

logger = logging.getLogger(__name__)

#: Column holding a scored row's model inputs, as a JSON object.
FEATURES_COLUMN = "features"


def feature_json(frame: pl.DataFrame, columns: Sequence[str]) -> pl.Expr:
    """Encode the named columns as one JSON object per row.

    Every position feeds its model a different column list, and those
    lists change, so the evaluation tables carry features as JSON rather
    than as a column each. DuckDB reads back into it with
    ``json_extract``.

    Parameters
    ----------
    frame : pl.DataFrame
        The frame the columns are read from. Only its schema is used.
    columns : Sequence[str]
        The model's feature names.

    Returns
    -------
    pl.Expr
        A string expression aliased to :data:`FEATURES_COLUMN`.
    """
    present = [column for column in columns if column in frame.columns]
    missing = sorted(set(columns) - set(present))
    if missing:
        logger.warning(
            "Scored rows are missing feature columns, so they will be "
            "absent from the stored JSON: %s",
            missing,
        )
    if not present:
        # ``pl.struct([])`` raises, which would surface as an opaque
        # polars error from inside training rather than as the warning
        # above naming the columns that went missing.
        return pl.lit("{}", dtype=pl.Utf8).alias(FEATURES_COLUMN)
    return pl.struct(present).struct.json_encode().alias(FEATURES_COLUMN)


def _fitted_feature_names(model: Pipeline) -> list[str] | None:
    """Return the feature names a fitted pipeline was trained on.

    Read from the first step, which is where sklearn records them and
    where the mismatch is raised from. None when the pipeline does not
    carry them, in which case there is nothing to compare.
    """
    for _, step in getattr(model, "steps", []):
        names = getattr(step, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return None


class ModelSpec(BaseModel):
    """Model spec for a predictor.

    ``position`` and ``component`` are what keep several models that
    share one output table apart. Every position's points model writes to
    ``points_component``, whose partition rewrites would otherwise delete
    the rows another position -- or another component of the same
    position -- had just written.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    registered_model_name: str
    production_alias: str
    table: Table
    #: Where this model's scored fold rows are kept for error analysis.
    evaluation_table: Table
    position: str | None = None
    #: Which scoring component this model predicts, where its table
    #: holds one row per component.
    component: str | None = None


class Predictor(ABC):
    """Abstract base class for predictor classes, with common methods."""

    def __init__(
        self,
        experiment_name: str,
        params: dict[str, Any],
        model_spec: ModelSpec,
        connection: duckdb.DuckDBPyConnection,
        fold_strategy: FoldStrategy,
    ) -> None:
        """Initialize the predictor.

        Parameters
        ----------
        experiment_name : str
            The name of the experiment to log to.
        params : dict[str, Any]
            The model parameters.
        model_spec : ModelSpec
            The model spec.
        connection : duckdb.DuckDBPyConnection
            The connection to the database.
        fold_strategy : FoldStrategy
            The fold strategy to use.
        """
        self.experiment_name = experiment_name
        self.params = params
        self.model_spec = model_spec
        self.connection = connection
        self.fold_strategy = fold_strategy
        # TODO(JT) - shall we just set this in the init?
        self._model_dataframe: pl.DataFrame | None = None
        self._scoring_dataframe: pl.DataFrame | None = None
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    @abstractmethod
    def build_training_data(self) -> pl.DataFrame:
        """Build the training dataset for the predictor.

        Method needs to be defined by the model inheriting from this class.
        This is always a historical dataset, where we know the target
        variable, allowing the model to be trained and evaluated upon.

        Returns
        -------
        pl.DataFrame
            The training dataset for the predictor.
        """
        ...

    @abstractmethod
    def build_forward_data(
        self, forward_fixtures: pl.DataFrame
    ) -> pl.DataFrame:
        """Build a dataset of feature values for forwards prediction.

        This method will build out all the features for the future
        gameweeks/fixtures (etc) that are being predicted upon in the future,
        allowing us to make predictions for the rest of the season.

        Parameters
        ----------
        forward_fixtures : pl.DataFrame
            The fixtures that we're going to predict in the future for.

        Returns
        -------
        pl.DataFrame
            The forward looking dataset with the training features.
        """
        ...

    @abstractmethod
    def fit_predict_fold(self, fold: Fold) -> FoldResult:
        """Fit a fresh model on one fold's train split and score its test.

        The only per-model part of cross validation. Implementations own
        their pipeline, their feature list and how predictions are turned
        into numbers -- a regressor calls ``predict``, the minutes
        classifier calls ``predict_proba`` -- and return whatever metrics
        container fits, so long as it satisfies :class:`Metrics`.

        The predictions travel with the metrics because they already
        exist at this point; recomputing them later would mean refitting
        the fold.

        Parameters
        ----------
        fold : Fold
            One fold, carrying non-empty train and test frames.

        Returns
        -------
        FoldResult
            The fold's scores and the rows they were computed over,
            shaped for the spec's evaluation table bar ``run_id``.
        """
        ...

    def cross_validate(
        self, folds: Sequence[Fold]
    ) -> tuple[list[FoldResult], dict[str, float]]:
        """Score the model over every fold and summarise the results.

        Folds with an empty side are skipped rather than scored: they
        carry no signal and would poison the aggregate with nans.

        Parameters
        ----------
        folds : Sequence[Fold]
            Folds from the injected :class:`FoldStrategy`.

        Returns
        -------
        tuple[list[FoldResult], dict[str, float]]
            Per-fold results in fold order, and the summary under the
            strategy's metric prefix. Both are empty when no fold has
            data on both sides.
        """
        per_fold = [
            self.fit_predict_fold(fold)
            for fold in folds
            if not fold.train.is_empty() and not fold.test.is_empty()
        ]
        return per_fold, summarise(
            [result.metrics for result in per_fold],
            self.fold_strategy.metric_prefix,
            self.fold_strategy.aggregates,
        )

    def store_fold_predictions(
        self, results: Sequence[FoldResult], run_id: str
    ) -> None:
        """Persist a run's scored rows for later error analysis.

        Appended rather than replaced: ``run_id`` is part of the
        evaluation table's primary key, so each run is its own partition
        and earlier runs stay readable beside it.

        Parameters
        ----------
        results : Sequence[FoldResult]
            Every scored fold, in fold order.
        run_id : str
            The MLflow run these predictions belong to.
        """
        frames = [
            result.predictions
            for result in results
            if not result.predictions.is_empty()
        ]
        if not frames:
            logger.info(
                "No scored rows for %s; storing no evaluation predictions.",
                self.model_spec.registered_model_name,
            )
            return
        rows = pl.concat(frames, how="vertical").with_columns(
            run_id=pl.lit(run_id)
        )
        self.model_spec.evaluation_table.append(self.connection, rows)

    @abstractmethod
    def train_final(self, feature_frame: pl.DataFrame) -> Pipeline:
        """Train the final model on the whole training dataset.

        Trains the final model on the whole training dataset, and returns the
        model object.

        Parameters
        ----------
        feature_frame : pl.DataFrame
            The training dataset to train the final model on.

        Returns
        -------
        Pipeline
            _description_
        """
        ...

    # TODO(JT): Make kind an enum
    # TODO(JT): Decouple feature frame as the name (just data?)
    @abstractmethod
    def build_prediction_rows(
        self,
        feature_frame: pl.DataFrame,
        model: Pipeline,
        version: str,
        kind: str,
    ) -> pl.DataFrame:
        """Store model predictions ready for storage in the database.

        Takes a feature dataset, trained model, model version and prediction
        kind and returns data.

        Parameters
        ----------
        feature_frame : pl.DataFrame
            A dataset of features to make predictions on.
        model : Pipeline
            The trained model to make predictions with.
        version : str
            The version of the model.
        kind : str
            The kind of prediction to make (e.g. backward or forward).

        Returns
        -------
        pl.DataFrame
            A dataset of predictions with model version and snapshot captured
            at time of prediction, for database storage.
        """
        ...

    def train_and_register_model(self) -> None:
        """Train a model, evaluate, and store in model registry.

        This method builds the training data for the model, creates the
        folds, scores the model over them and then trains the final model
        itself. All metrics are logged to MLFlow for each fold and the
        aggregate metrics at the end too.

        The run is opened before the folds are scored, because the stored
        predictions are keyed on its id.

        The final model is fitted on the whole frame, holdout included.
        The score therefore estimates the training procedure rather than
        the exact registered artifact, which is the trade that lets the
        deployed model see the most recent gameweeks.
        """
        if self._model_dataframe is None:
            self._model_dataframe = self.build_training_data()
        folds = list(self.fold_strategy.split(self._model_dataframe))
        mlflow.set_experiment(self.experiment_name)
        prefix = self.fold_strategy.metric_prefix
        with mlflow.start_run() as run:
            per_fold, agg = self.cross_validate(folds)
            final_model = self.train_final(self._model_dataframe)
            mlflow.log_params(self.params)
            for step, result in enumerate(per_fold):
                for key, value in result.metrics.as_dict().items():
                    mlflow.log_metric(f"{prefix}_{key}", value, step=step)
            mlflow.log_metrics(agg)
            mlflow.sklearn.log_model(
                final_model,
                name="model",
                registered_model_name=self.model_spec.registered_model_name,
            )
            # Last, so a run that dies while fitting or registering
            # leaves no evaluation rows behind. DuckDB commits on write,
            # and rows keyed to a failed run are indistinguishable from
            # good ones at query time.
            self.store_fold_predictions(per_fold, run.info.run_id)

        logger.info(
            f"Model {self.model_spec.registered_model_name} trained and registered"
        )

    @property
    def expected_features(self) -> list[str] | None:
        """Return the feature names this model's code declares.

        None means "do not check", which is the right answer for a
        predictor whose inputs are not a flat named list.
        """
        return None

    def production_model(self) -> tuple[str, Pipeline] | None:
        """Load the aliased model, refusing one the code has outgrown.

        The registry and the code drift apart on their own: the alias is
        moved by hand, so a model fitted on an older feature list stays
        live until someone promotes a newer one. Scoring with it fails
        six frames inside sklearn on a message that names a column and
        nothing else -- not the model, not the alias, not the remedy.

        Returns
        -------
        tuple[str, Pipeline] | None
            The live version and model, or None when there is no alias or
            the aliased model disagrees with the code about its inputs.
        """
        production = load_production_model(
            self.model_spec.registered_model_name,
            self.model_spec.production_alias,
        )
        if production is None:
            logger.warning(
                "No %s alias on %s; skipping.",
                self.model_spec.production_alias,
                self.model_spec.registered_model_name,
            )
            return None
        version, model = production
        declared = self.expected_features
        fitted = _fitted_feature_names(model)
        if declared is None or fitted is None:
            return version, model
        surplus = sorted(set(fitted) - set(declared))
        absent = sorted(set(declared) - set(fitted))
        if surplus or absent:
            logger.warning(
                "%s version %s is aliased %s but was fitted on different "
                "features, so it cannot score the current frame. It wants "
                "%s that the code no longer builds, and does not know about "
                "%s. Train a version on the current code and move the alias "
                "onto it in the MLflow UI.",
                self.model_spec.registered_model_name,
                version,
                self.model_spec.production_alias,
                surplus or "nothing",
                absent or "nothing",
            )
            return None
        return version, model

    @property
    def served_positions(self) -> tuple[str, ...] | str:
        """Return the position, or positions, this model writes rows for.

        The spec's single position by default. Subclasses that serve
        several override it; the storage layer reads a tuple as an
        ``IN`` predicate.
        """
        return self.model_spec.position or ()

    @property
    def _own_rows(self) -> dict[str, object]:
        """Predicates isolating this model's rows in a shared table.

        Empty when the spec names neither a position nor a component,
        which is the right answer for a model that owns its output table
        outright.
        """
        own: dict[str, object] = {}
        if self.model_spec.position is not None:
            # Every position this model writes, not just its primary
            # one. Deleting one position's rows and then inserting all
            # of them would leave the rest stored twice.
            own["position"] = self.served_positions
        if self.model_spec.component is not None:
            own["component"] = self.model_spec.component
        return own

    # TODO(JT): Does this need to be a method?
    def get_prediction_versions(self, seasons: list[str]) -> set[str]:
        """Get model versions that have been used for previous prediction.

        Restricted to this model's own rows, so a position sharing
        ``points_prediction`` with the other positions does not read
        their versions as its own.

        Parameters
        ----------
        seasons : list[str]
            A list of seasons to check the model version against

        Returns
        -------
        set[str]
            The model versions that have been used
        """
        return prediction_versions(
            self.connection,
            self.model_spec.table.name,
            seasons,
            equals=self._own_rows,
        )

    # TODO(JT): Will there be issues with partial seasons here?
    def scoring_frame(self) -> pl.DataFrame:
        """Return the rows to backfill predictions for.

        The training frame by default, which is right for a model whose
        training population is the population it serves. It is not right
        for one that restricts its fit -- a row dropped for being a
        fifteen-minute cameo still has to be scored, and a model pooled
        across positions must not write rows for the positions it does
        not serve.

        Returns
        -------
        pl.DataFrame
            Rows shaped like the training frame.
        """
        if self._model_dataframe is None:
            self._model_dataframe = self.build_training_data()
        return self._model_dataframe

    def predict_backwards(
        self, season: str, model: Pipeline, version: str
    ) -> None:
        """Use the trained model to backfill predictions for a season.

        Parameters
        ----------
        season: str
            The season to backfill predictions for
        model: Pipeline
            The trained model
        version: str
            The model version
        """
        sub = self.scoring_frame().filter(pl.col("season") == season)
        if sub.is_empty():
            logger.warning(
                f"No training data for season {season}; skipping backwards prediction."
            )
            return
        rows = self.build_prediction_rows(sub, model, version, BACKFILL_KIND)
        self.model_spec.table.replace_partition(
            self.connection,
            rows,
            equals={
                "season": season,
                "prediction_kind": BACKFILL_KIND,
                **self._own_rows,
            },
        )

    # TODO(JT): We need to backfill predictions for partial seasons
    # I think it might do this already?
    def backfill_model_predictions(self) -> None:
        """Backfill model predictions for all seasons in training data.

        This will always run the predict_backwards method for the current
        season first.

        It then looks at all available seasons in the training data (except
        the current one), and filters the training dataset for this. It then
        loads the production model. If the production model has already
        been used to make predictions on previous data, we don't run it.
        Otherwise, for each season, we run the predict_backwards method.
        """
        if self._model_dataframe is None:
            self._model_dataframe = self.build_training_data()

        all_seasons = set(self._model_dataframe["season"].unique().to_list())
        historic_seasons = sorted(all_seasons - {CURRENT_SEASON})
        production = self.production_model()
        if production is None:
            logger.warning(
                f"No usable production model for {self.model_spec.registered_model_name}; skipping backwards prediction."
            )
            return
        production_version, model = production
        stored_versions = self.get_prediction_versions(historic_seasons)
        stored_seasons = self.model_spec.table.seasons_present(
            self.connection, equals=self._own_rows
        ) & set(historic_seasons)

        self.predict_backwards(CURRENT_SEASON, model, production_version)
        # TODO (JT): Make this logc clearer
        historic_needs_rebuild = bool(historic_seasons) and (
            stored_versions != {production_version}
            or stored_seasons != set(historic_seasons)
        )

        if historic_needs_rebuild:
            for season in historic_seasons:
                self.predict_backwards(season, model, production_version)
            logger.info(
                f"Backfilled historic {self.model_spec.table.name} predictions for {historic_seasons} at version {production_version}"
            )
        else:
            logger.info(
                f"Historic {self.model_spec.table.name} predictions already at version {production_version}; skipping."
            )

    def predict_forward(self) -> None:
        """Use the trained model to make future predictions.

        This loads the production model, and builds the forward looking
        feature dataset for future fixtures in the current season. Afterwards,
        it builds the predictions and stores them in the database.
        """
        production = self.production_model()
        if production is None:
            logger.warning(
                f"No usable production model for {self.model_spec.registered_model_name}; skipping forward prediction."
            )
            return
        production_version, model = production
        snapshot = latest_snapshot(
            PLAYER_SNAPSHOT.load(self.connection), CURRENT_SEASON
        )
        if snapshot.is_empty():
            logger.warning(
                f"No player snapshot for {CURRENT_SEASON}; skipping forward scoring."
            )
            return
        player_week = PLAYER_WEEK.load(self.connection)
        player_season = PLAYER_SEASON.load(self.connection)
        warn_unidentified_snapshot(snapshot, player_season, CURRENT_SEASON)
        from_gw = last_played_gw(player_week, CURRENT_SEASON) + 1
        team_name_to_id = {team.name: team.id for team in FplAPI().get_teams()}
        forward_fixtures = build_forward_fixtures(
            snapshot,
            TEAM_FIXTURE.load(self.connection),
            CURRENT_SEASON,
            from_gw,
            team_name_to_id,
        )
        if forward_fixtures.is_empty():
            logger.info(
                f"No unplayed {CURRENT_SEASON} fixtures from gw {from_gw}; nothing to score."
            )
            return
        forward_data = self.build_forward_data(forward_fixtures)
        rows = self.build_prediction_rows(
            forward_data, model, production_version, FORWARD_KIND
        )
        self.model_spec.table.replace_partition(
            self.connection,
            rows,
            equals={
                "season": CURRENT_SEASON,
                "prediction_kind": FORWARD_KIND,
                **self._own_rows,
            },
            gw_from=from_gw,
        )
