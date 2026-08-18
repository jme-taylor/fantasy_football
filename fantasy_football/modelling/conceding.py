"""The team conceding head, behind clean sheets and the deduction.

Goals conceded is a team's quantity. Eleven players share one clean
sheet, and the per-player variation in what they were on the pitch for
is substitution timing, not defending. So this head trains on one row
per team-fixture and the fan-out puts the same prediction on every
defender of that team -- where a player-grain fit would model the same
number eleven times over and read the duplication as evidence.

**Per match, not per 90.** Every sibling head predicts a rate and lets
minutes scale it. A team fixture is always ninety minutes, so the
prediction here is already a match-level expected count: no weight
column, no per-90 division, no minutes floor on the target. Minutes
re-enter once, in :class:`ConcedingComponent`, and only on the deduction
leg.

The target reads FCI's own team counter rather than summing player
goals, because an own goal is conceded by a team and credited to no
player. FCI covers 2024-25 onwards including the live season, so unlike
the goals and assists heads there is no second source to fall back to
and no coalesce -- its absence here is deliberate, not an oversight.

``team_match`` is built from ``opta_match``, so a club sitting in
``team_fixture`` without match data -- the known 2025-26 contamination --
produces no training row at all. The guard is the derivation, not a
filter.

What the component pays is *not* what this head targets, and that is
the first time a carve-out has broken the "one definition, two
aliasings" rule. It has to: this head predicts a team's goals, while
FPL pays a player for his own on-pitch clean sheet. The invariant that
still holds is that :func:`conceding_points_sql` and
:class:`ConcedingComponent` describe the same *rule*; the team rate is
an intermediate quantity, not a component.
"""

import logging
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import ClassVar, override

import numpy as np
import polars as pl
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from fantasy_football.constants import ROLLING_WINDOW
from fantasy_football.features.team_form import measure_column_name
from fantasy_football.features.views import register_feature_views
from fantasy_football.modelling.components import (
    CLEAN_SHEET_POINTS_BY_POSITION,
    CONCEDING_DEDUCTION_POSITIONS,
    KEY_COLUMNS,
    PREDICTED_VALUE,
    Component,
    ConcedingComponent,
    expected_deduction,
)
from fantasy_football.modelling.defcon import (
    MINUTES_FLOOR as DEFCON_MINUTES_FLOOR,
)
from fantasy_football.modelling.distributions import PoissonCounts
from fantasy_football.modelling.folds import Fold, FoldResult
from fantasy_football.modelling.metrics import count_poisson_deviance
from fantasy_football.modelling.points import asof_form
from fantasy_football.modelling.predictor import (
    ModelSpec,
    Predictor,
    feature_json,
)
from fantasy_football.storage.coverage import FCI_SEASONS
from fantasy_football.storage.tables import (
    MINUTES_PREDICTION,
    POINTS_COMPONENT,
    TEAM_FIXTURE,
    TEST_CONCEDING_PREDICTION,
)

logger = logging.getLogger(__name__)

#: The position this instance fans out to and writes. The model behind
#: it knows nothing about positions at all.
POSITION = "DEF"

REGISTERED_MODEL = "team_conceding_regressor"
PRODUCTION_ALIAS = "production"
EXPERIMENT_NAME = "team-conceding-model"

#: Seasons with the team counters and expected-goals measures this reads.
TRAINING_SEASONS = FCI_SEASONS

#: Matches a team's rolling window must span before this head will learn
#: from the fixture. Below it the features are an average of one or two
#: matches, which teaches the model to trust noise. Those rows are still
#: scored -- an opening gameweek needs a prediction -- and there they
#: take the imputed median without any flag saying so, because the team
#: form view has no no-form column the way the player one does.
FORM_MATCHES_FLOOR = ROLLING_WINDOW

#: Matched to the sibling DEF components, and nothing tighter. A leg
#: missing one component is dropped from the composed prediction
#: entirely, so a stricter floor here would delete predictions the other
#: four were happy to make.
SCORING_MINUTES_FLOOR = DEFCON_MINUTES_FLOOR

#: Team-fixture grain, as against ``components.KEY_COLUMNS``.
TEAM_KEY_COLUMNS = ["season", "gw", "team", "opposition"]

TARGET = "goals_conceded"

_OWN_MEASURES = ("xg_against", "goals_against", "clean_sheet")
_OPPOSITION_MEASURES = ("xg_for", "goals_for")


def _own_feature_names() -> list[str]:
    """Return the rolling columns describing the team's own defence."""
    return [
        measure_column_name(measure, ROLLING_WINDOW)
        for measure in _OWN_MEASURES
    ]


def _opposition_feature_names() -> list[str]:
    """Return the rolling columns describing the opponent's attack.

    Prefixed, unlike the defender model's, because both sides of the
    fixture are read off the same view here and an unprefixed
    ``goals_for_rolling_5`` would not say whose.
    """
    return [
        f"opposition_{measure_column_name(measure, ROLLING_WINDOW)}"
        for measure in _OPPOSITION_MEASURES
    ]


FEATURES = ["is_home", *_own_feature_names(), *_opposition_feature_names()]


def conceding_points_sql(position: str, fpl: str = "pmf") -> str:
    """Return the conceding points FPL paid a player, as SQL.

    What the component pays. Reads the player's own settled figures
    rather than the team's, because the payment is a player's: FPL's
    ``clean_sheets`` already carries the hour requirement, and
    ``goals_conceded`` counts only what was shipped while he was on.

    Parameters
    ----------
    position : str
        The position being priced.
    fpl : str, optional
        Alias of the ``player_match_fpl`` relation.

    Returns
    -------
    str
        A scalar expression, unaliased.
    """
    clean_sheet = (
        f"{CLEAN_SHEET_POINTS_BY_POSITION[position]} "
        f"* coalesce({fpl}.clean_sheets, 0)"
    )
    if position not in CONCEDING_DEDUCTION_POSITIONS:
        return f"({clean_sheet})"
    return (
        f"({clean_sheet} " f"- floor(coalesce({fpl}.goals_conceded, 0) / 2))"
    )


@dataclass(frozen=True, slots=True)
class ConcedingMetrics:
    """Scores for the conceding head over one fold.

    Both payoffs are scored, separately. They are wrong in different
    ways under a Poisson -- over-dispersion raises the clean-sheet
    chance and the deduction together -- so a single count-level metric
    would hide which leg is drifting.

    ``calibration`` is mean predicted over mean actual: above one is a
    head expecting more goals than the league concedes.
    """

    poisson_deviance: float
    mae: float
    calibration: float
    clean_sheet_brier: float
    base_rate_brier: float
    skill_score: float
    clean_sheet_rate: float
    deduction_mae: float

    def as_dict(self) -> dict[str, float]:
        """Return the metric names and values, one level deep."""
        return asdict(self)


class ConcedingPredictor(Predictor):
    """Predicts a team's goals conceded in a fixture, per match."""

    POSITION = POSITION
    TARGET = TARGET
    COMPONENT = Component.CONCEDING
    COMPONENT_IMPL: ClassVar[ConcedingComponent] = ConcedingComponent()
    TRAINING_SEASONS = TRAINING_SEASONS
    FEATURES: ClassVar[list[str]] = FEATURES

    @property
    @override
    def expected_features(self) -> list[str]:
        """Return the feature names this head declares."""
        return list(self.FEATURES)

    @property
    def _seasons_predicate(self) -> str:
        """Return the season filter both frames share."""
        named = ", ".join(f"'{season}'" for season in self.TRAINING_SEASONS)
        return f"IN ({named})"

    def training_frame_sql(self) -> str:
        """Return the SELECT behind the team-grain training frame.

        Both sides of a fixture share a ``match_id``, which is what lets
        one view supply a team's own defensive form and its opponent's
        attacking form without resolving anything by name.
        """
        own = ",\n    ".join(
            f"own.{column} AS {column}" for column in _own_feature_names()
        )
        opposition = ",\n    ".join(
            f"opp.{column} AS {alias}"
            for column, alias in zip(
                [
                    measure_column_name(measure, ROLLING_WINDOW)
                    for measure in _OPPOSITION_MEASURES
                ],
                _opposition_feature_names(),
                strict=True,
            )
        )
        return f"""
SELECT
    own.season,
    own.gw,
    own.team,
    own.opposition,
    own.goals_against AS {TARGET},
    CAST(own.is_home AS DOUBLE) AS is_home,
    {own},
    {opposition}
FROM team_match_form AS own
LEFT JOIN team_match_form AS opp
    ON  opp.season   = own.season
    AND opp.gw       = own.gw
    AND opp.match_id = own.match_id
    AND opp.team     = own.opposition
WHERE own.season {self._seasons_predicate}
  AND own.goals_against IS NOT NULL
  AND own.form_matches >= {FORM_MATCHES_FLOOR}
"""

    def scoring_frame_sql(self) -> str:
        """Return the SELECT fanning a team's fixture out to its players.

        Joined on the opponent and the kickoff rather than on the
        player's own club. A club read off a snapshot is the club he
        ended the season at, and the per-gameweek source has thousands of
        null-team rows, so resolving the player's side would lose
        defenders this head is meant to score.

        The kickoff is load-bearing and not decoration. ``(season, gw,
        opposition)`` alone is ambiguous in a double gameweek: a team
        playing twice is the opponent of two different clubs that week,
        so both of their fixtures would match and every defender would
        fan out twice.
        """
        own = ",\n    ".join(
            f"own.{column} AS {column}" for column in _own_feature_names()
        )
        opposition = ",\n    ".join(
            f"opp.{column} AS {alias}"
            for column, alias in zip(
                [
                    measure_column_name(measure, ROLLING_WINDOW)
                    for measure in _OPPOSITION_MEASURES
                ],
                _opposition_feature_names(),
                strict=True,
            )
        )
        return f"""
SELECT
    m.season,
    m.gw,
    m.element,
    m.opponent,
    CAST(m.is_home AS DOUBLE) AS is_home,
    {own},
    {opposition}
FROM player_match AS m
INNER JOIN player_season AS s
    ON  m.element  = s.element
    AND m.season   = s.season
    AND s.position = '{self.POSITION}'
LEFT JOIN fpl_team_id AS opp_id
    ON  opp_id.season  = m.season
    AND opp_id.team_id = m.opponent
LEFT JOIN team_match_form AS own
    ON  own.season       = m.season
    AND own.gw           = m.gw
    AND own.opposition   = opp_id.team
    AND own.kickoff_time = m.kickoff_time
LEFT JOIN team_match_form AS opp
    ON  opp.season   = own.season
    AND opp.gw       = own.gw
    AND opp.match_id = own.match_id
    AND opp.team     = own.opposition
WHERE m.minutes IS NOT NULL
  AND m.season {self._seasons_predicate}
  AND m.minutes >= {SCORING_MINUTES_FLOOR}
"""

    @override
    def build_training_data(self) -> pl.DataFrame:
        """Build the team-grain training frame from the store."""
        register_feature_views(self.connection)
        frame = self.connection.sql(self.training_frame_sql()).pl()
        return frame.select(TEAM_KEY_COLUMNS + [self.TARGET] + self.FEATURES)

    @override
    def scoring_frame(self) -> pl.DataFrame:
        """Return every defender leg this head must write a component for.

        Player grain, unlike the training frame. The base class documents
        this hook as the seam for a model whose training population is
        not the population it serves, and a team head is the strongest
        case of that there is.
        """
        if self._scoring_dataframe is None:
            register_feature_views(self.connection)
            self._scoring_dataframe = (
                self.connection.sql(self.scoring_frame_sql())
                .pl()
                .select(KEY_COLUMNS + self.FEATURES)
            )
        return self._scoring_dataframe

    @override
    def build_forward_data(
        self, forward_fixtures: pl.DataFrame
    ) -> pl.DataFrame:
        """Build features for this position's unplayed fixtures.

        The team fixtures come out of ``forward_fixtures`` rather than
        off ``team_fixture`` directly, which is what keeps a contaminated
        club from re-entering on this path: a fixture with no rostered
        players is not in there to begin with.

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
        register_feature_views(self.connection)
        forward_fixtures = forward_fixtures.filter(
            pl.col("position") == self.POSITION
        )
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
        frame = forward_fixtures.join(
            fixtures, on=["season", "gw", "team", "kickoff_time"], how="left"
        )
        frame = asof_form(frame, team_form, ["team"], _own_feature_names())
        opposition = team_form.select(
            pl.col("team").alias("opponent_name"),
            "kickoff_time",
            *[
                pl.col(measure_column_name(measure, ROLLING_WINDOW)).alias(
                    alias
                )
                for measure, alias in zip(
                    _OPPOSITION_MEASURES,
                    _opposition_feature_names(),
                    strict=True,
                )
            ],
        )
        frame = asof_form(
            frame,
            opposition,
            ["opponent_name"],
            _opposition_feature_names(),
        )
        frame = frame.with_columns(pl.col("is_home").cast(pl.Float64))
        missing = [name for name in self.FEATURES if name not in frame.columns]
        if missing:
            frame = frame.with_columns(
                [
                    pl.lit(None, dtype=pl.Float64).alias(name)
                    for name in missing
                ]
            )
            logger.warning(
                "Forward conceding features absent from the join: %s", missing
            )
        return frame.select(KEY_COLUMNS + self.FEATURES)

    def make_pipeline(self) -> Pipeline:
        """Build the median-imputing Poisson-loss pipeline.

        A boosted Poisson rather than the squared-error forest every
        other head uses. The target is a count, the output is a rate fed
        straight into a Poisson, and a squared-error fit neither
        guarantees a non-negative prediction nor optimises the
        likelihood the component then assumes.

        Returns
        -------
        sklearn.pipeline.Pipeline
            Unfitted pipeline ending in a Poisson regressor.
        """
        return Pipeline(
            [
                ("impute", SimpleImputer(strategy="median")),
                ("model", HistGradientBoostingRegressor(loss="poisson")),
            ]
        )

    def fold_metrics(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> ConcedingMetrics:
        """Score the fold on both payoffs, not only on the count."""
        distribution = PoissonCounts()
        rate = np.clip(np.asarray(predicted, dtype=float), 0.0, None)
        actual = test_df[self.TARGET].cast(pl.Float64).to_numpy()
        kept = (actual == 0).astype(int)
        clean = 1.0 - distribution.p_at_least(rate, 1)
        base_rate = float(np.mean(kept))
        brier = float(np.mean((clean - kept) ** 2))
        base_brier = float(np.mean((base_rate - kept) ** 2))
        mean_actual = float(np.mean(actual))
        return ConcedingMetrics(
            poisson_deviance=count_poisson_deviance(actual, rate),
            mae=float(np.mean(np.abs(rate - actual))),
            calibration=(
                float(np.mean(rate)) / mean_actual if mean_actual else 0.0
            ),
            clean_sheet_brier=brier,
            base_rate_brier=base_brier,
            skill_score=1.0 - brier / base_brier if base_brier else 0.0,
            clean_sheet_rate=base_rate,
            deduction_mae=float(
                np.mean(
                    np.abs(
                        expected_deduction(rate, distribution)
                        - np.floor(actual / 2.0)
                    )
                )
            ),
        )

    def fold_predictions(
        self, test_df: pl.DataFrame, predicted: Sequence[float]
    ) -> pl.DataFrame:
        """Shape one fold's scored rows for the team-grain eval table."""
        return test_df.with_columns(
            predicted_conceded=pl.Series(predicted).cast(pl.Float64),
            actual_conceded=pl.col(self.TARGET).cast(pl.Float64),
            features=feature_json(test_df, self.FEATURES),
        ).select(
            TEAM_KEY_COLUMNS
            + ["predicted_conceded", "actual_conceded", "features"]
        )

    @override
    def fit_predict_fold(self, fold: Fold) -> FoldResult:
        """Fit on the fold's train split and score its test split."""
        pipe = self.make_pipeline()
        pipe.fit(
            fold.train.select(self.FEATURES).to_pandas(),
            fold.train[self.TARGET].to_list(),
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
        )
        return pipe

    def minutes_predictions(self, kind: str) -> pl.DataFrame:
        """Return the minutes forecast of one kind."""
        return MINUTES_PREDICTION.load(self.connection).filter(
            pl.col("prediction_kind") == kind
        )

    @override
    def build_prediction_rows(
        self,
        feature_frame: pl.DataFrame,
        model: Pipeline,
        version: str,
        kind: str,
    ) -> pl.DataFrame:
        """Score player rows and shape them as the conceding component."""
        predicted = model.predict(
            feature_frame.select(self.FEATURES).to_pandas()
        )
        scored = feature_frame.select(KEY_COLUMNS).with_columns(
            position=pl.lit(self.POSITION),
            **{PREDICTED_VALUE: pl.Series(predicted).cast(pl.Float64)},
        )
        rows = self.COMPONENT_IMPL.points(
            scored, self.minutes_predictions(kind), kind
        )
        return rows.with_columns(model_version=pl.lit(version)).select(
            POINTS_COMPONENT.columns
        )


MID_POSITION = "MID"


class MidfielderConcedingPredictor(ConcedingPredictor):
    """The same team-grain model, fanned out to midfielders.

    A separate instance rather than a serving list: the model predicts
    goals conceded by a *team* and carries no position anywhere in its
    features or its target, so there is nothing here for a position
    dummy to tell apart. It reads the same registered model and the same
    alias -- only the rows it writes differ, which the spec's position
    keeps apart.
    """

    POSITION = MID_POSITION


CONCEDING_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_CONCEDING_PREDICTION,
    position=POSITION,
    component=Component.CONCEDING,
)


MIDFIELDER_CONCEDING_SPEC = ModelSpec(
    registered_model_name=REGISTERED_MODEL,
    production_alias=PRODUCTION_ALIAS,
    table=POINTS_COMPONENT,
    evaluation_table=TEST_CONCEDING_PREDICTION,
    position=MID_POSITION,
    component=Component.CONCEDING,
)
