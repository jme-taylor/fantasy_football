"""Train and register the minutes model on its own.

    uv run python scripts/train_minutes.py

Trains, scores the holdout, logs the run and registers a version in
MLflow. It writes **no predictions**: the pipeline in ``main.py`` stays
the only thing that writes ``minutes_prediction``, and it can only do so
once a version is live.

This exists for the cutover. ``main.py`` trains, backfills and predicts
forward in one pass, so on a run where the feature set has changed the
backfill loads an alias the code has outgrown, logs a skip and leaves
every downstream model without minutes. Registering here first, promoting
by hand, then running ``main.py`` avoids that window.

Promotion is by hand: move the ``production`` alias in the MLflow UI once
a run looks better than the live one.
"""

import logging

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.logging_config import configure_logging
from fantasy_football.modelling.folds import TrainTestSplitStrategy
from fantasy_football.modelling.minutes import MINUTES_SPEC, MinutesPredictor
from fantasy_football.storage.database import get_connection

logger = logging.getLogger(__name__)

EXPERIMENT_NAME = "minutes_played_classification"


def main() -> None:
    """Train the minutes model and log the run to MLflow."""
    configure_logging()
    connection = get_connection(DATABASE_PATH)
    try:
        predictor = MinutesPredictor(
            experiment_name=EXPERIMENT_NAME,
            params={},
            model_spec=MINUTES_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(),
        )
        predictor.train_and_register_model()
    finally:
        connection.close()
    logger.info(
        "Registered a new %s version. Move the %s alias in the MLflow UI "
        "before running main.py.",
        MINUTES_SPEC.registered_model_name,
        MINUTES_SPEC.production_alias,
    )


if __name__ == "__main__":
    main()
