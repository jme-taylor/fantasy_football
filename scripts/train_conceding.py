"""Train and evaluate the team conceding head on its own.

    uv run python scripts/train_conceding.py

Trains, cross-validates, logs the metrics and registers a version in
MLflow. It writes **no predictions**, for the same reason
``train_goals.py`` does not: a second writer into ``points_component``
would leave the stored components summing to something other than the
stored total depending on which ran last, so the pipeline in ``main.py``
stays the only thing that writes them.

Promotion is by hand: move the ``production`` alias in the MLflow
UI once a run looks better than the live one.
"""

import logging

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.logging_config import configure_logging
from fantasy_football.modelling.conceding import (
    CONCEDING_SPEC,
    EXPERIMENT_NAME,
    TRAINING_SEASONS,
    ConcedingPredictor,
)
from fantasy_football.modelling.folds import TrainTestSplitStrategy
from fantasy_football.storage.database import get_connection

logger = logging.getLogger(__name__)


def main() -> None:
    """Train the team conceding head and log the run to MLflow."""
    configure_logging()
    connection = get_connection(DATABASE_PATH)
    try:
        predictor = ConcedingPredictor(
            experiment_name=EXPERIMENT_NAME,
            params={"training_seasons": ",".join(TRAINING_SEASONS)},
            model_spec=CONCEDING_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=TRAINING_SEASONS
            ),
        )
        predictor.train_and_register_model()
        logger.info(
            "Registered a new %s version. Promote it in the MLflow UI to "
            "put it live; nothing has been written to "
            "points_component.",
            CONCEDING_SPEC.registered_model_name,
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
