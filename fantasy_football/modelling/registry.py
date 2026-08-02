"""Loading the alias-promoted model out of the MLflow Model Registry.

Every training run registers a new version. Which version is live is
decided by an alias moved by hand in the MLflow UI, so "no alias yet" is
a normal state before the first manual promotion, not an error. This
returns None for it and lets the caller decide whether that is fatal.
"""

import logging
from typing import Any

import mlflow
import mlflow.sklearn
import mlflow.tracking
from mlflow.exceptions import MlflowException

from fantasy_football.constants import MLFLOW_TRACKING_URI

logger = logging.getLogger(__name__)


def load_production_model(
    registered_name: str, alias: str
) -> tuple[str, Any] | None:
    """Return the aliased version string and loaded model, or None.

    Parameters
    ----------
    registered_name : str
        Name of the registered model in the MLflow Model Registry.
    alias : str
        Alias identifying the live version, e.g. ``"production"``.

    Returns
    -------
    tuple[str, Any] | None
        The version string and the loaded model, or ``None`` when the
        registered model or alias does not exist.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    client = mlflow.tracking.MlflowClient()
    try:
        version = client.get_model_version_by_alias(registered_name, alias)
    except MlflowException:
        logger.warning(
            "No %s alias on %s; promote a version in the MLflow UI first.",
            alias,
            registered_name,
        )
        return None
    model = mlflow.sklearn.load_model(f"models:/{registered_name}@{alias}")
    return version.version, model
