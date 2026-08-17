"""Train and evaluate the pooled yellow-cards rate head on its own.

    uv run python scripts/train_yellow_cards.py

Trains, cross-validates, logs the match-scale metrics and registers a
version in MLflow. It writes **no predictions**, for the same reason
``train_assists.py`` does not: a second writer into ``points_component``
would leave the stored components summing to something other than the
stored total depending on which ran last, so the pipeline in ``main.py``
stays the only thing that writes them.

Promotion is by hand, and the order matters. This head and the defender
residual head are a matched pair -- the residual's target deducts the
bookings this one predicts -- so promote the retrained residual *first*.
Between the two moves defenders are charged slightly too little for
their cards, which is recoverable. The other order charges every booking
twice, which is silent.

The window is narrower than it was for conceding: a yellow is one point
against a clean sheet's four, and it lands on a minority of legs.
"""

import logging

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.logging_config import configure_logging
from fantasy_football.modelling.folds import TrainTestSplitStrategy
from fantasy_football.modelling.yellow_cards import (
    EXPERIMENT_NAME,
    TEST_SEASONS,
    TRAINING_POSITIONS,
    TRAINING_SEASONS,
    YELLOW_CARDS_SPEC,
    YellowCardsRatePredictor,
)
from fantasy_football.storage.database import get_connection

logger = logging.getLogger(__name__)


def main() -> None:
    """Train the yellow-cards rate head and log the run to MLflow."""
    configure_logging()
    connection = get_connection(DATABASE_PATH)
    try:
        predictor = YellowCardsRatePredictor(
            experiment_name=EXPERIMENT_NAME,
            params={
                "training_seasons": ",".join(TRAINING_SEASONS),
                "training_positions": ",".join(TRAINING_POSITIONS),
                "test_seasons": ",".join(TEST_SEASONS),
            },
            model_spec=YELLOW_CARDS_SPEC,
            connection=connection,
            fold_strategy=TrainTestSplitStrategy(test_seasons=TEST_SEASONS),
        )
        predictor.train_and_register_model()
        logger.info(
            "Registered a new %s version. Promote the retrained defender "
            "residual before this one; nothing has been written to "
            "points_component.",
            YELLOW_CARDS_SPEC.registered_model_name,
        )
    finally:
        connection.close()


if __name__ == "__main__":
    main()
