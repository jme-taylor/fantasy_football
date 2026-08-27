"""Rank the scoring components by how much each one's error costs.

    uv run python scripts/component_ablation.py

Replaces one component at a time with what actually happened, holding the
rest at their held-out predictions, and reports how far the composed total
improves. The component with the largest gain is where modelling effort pays
back most -- which is not the component with the largest error.

Reads the evaluation tables only. Nothing is trained, and nothing is written
to the database or to MLflow.
"""

import logging

from fantasy_football.constants import DATABASE_PATH
from fantasy_football.logging_config import configure_logging
from fantasy_football.modelling.ablation import run_ablation, summarise
from fantasy_football.storage.database import get_connection

logger = logging.getLogger(__name__)


def main() -> None:
    """Run the ablation and print the leverage table per position."""
    configure_logging()
    connection = get_connection(DATABASE_PATH)
    try:
        results = run_ablation(connection)
    finally:
        connection.close()
    print(summarise(results))  # noqa: T201


if __name__ == "__main__":
    main()
