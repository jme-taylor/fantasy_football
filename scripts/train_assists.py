"""Train and evaluate the pooled assists rate head on its own.

    uv run python scripts/train_assists.py

Trains, cross-validates, logs the match-scale metrics and registers a
version in MLflow. It writes **no predictions**, for the same reason
``train_goals.py`` does not: a second writer into ``points_component``
would leave the stored components summing to something other than the
stored total depending on which ran last, so the pipeline in ``main.py``
stays the only thing that writes them.

Promotion is by hand, and the order matters. This head and the defender
residual head are a matched pair -- the residual's target deducts the
assists this one predicts -- so promote the retrained residual *first*.
Between the two moves the defender prediction is slightly low, which is
recoverable. The other order double-counts every assist, which is
silent.
"""

import logging

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.logging_config import configure_logging
from fantasy_football.modelling.assists import (
    ASSISTS_SPEC,
    EXPERIMENT_NAME,
    TRAINING_POSITIONS,
    TRAINING_SEASONS,
    AssistsRatePredictor,
)
from fantasy_football.modelling.folds import TrainTestSplitStrategy
from fantasy_football.storage.database import get_connection

logger = logging.getLogger(__name__)


def main() -> None:
    """Train the assists rate head and log the run to MLflow."""
    configure_logging()
    connection = get_connection(DATABASE_PATH)
    try:
        predictor = AssistsRatePredictor(
            experiment_name=EXPERIMENT_NAME,
            params={
                "training_seasons": ",".join(TRAINING_SEASONS),
                "training_positions": ",".join(TRAINING_POSITIONS),
            },
            model_spec=ASSISTS_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(
                test_seasons=TRAINING_SEASONS
            ),
        )
        predictor.train_and_register_model()
        logger.info(
            "Registered a new %s version. Promote the retrained defender "
            "residual before this one; nothing has been written to "
            "points_component.",
            ASSISTS_SPEC.registered_model_name,
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
