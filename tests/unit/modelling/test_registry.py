from mlflow.exceptions import MlflowException

from fantasy_football.modelling import registry


def test_returns_version_and_model_when_alias_exists(mocker):
    """Return the version string and loaded model when the alias exists."""
    version = mocker.Mock()
    version.version = "7"
    client = mocker.Mock()
    client.get_model_version_by_alias.return_value = version
    mocker.patch.object(registry.mlflow, "set_tracking_uri")
    mocker.patch.object(
        registry.mlflow.tracking, "MlflowClient", return_value=client
    )
    load = mocker.patch.object(
        registry.mlflow.sklearn, "load_model", return_value="MODEL"
    )

    assert registry.load_production_model("m", "production") == ("7", "MODEL")
    load.assert_called_once_with("models:/m@production")


def test_returns_none_when_no_alias(mocker, caplog):
    """Return None and log a warning when the alias does not exist."""
    client = mocker.Mock()
    client.get_model_version_by_alias.side_effect = MlflowException("nope")
    mocker.patch.object(registry.mlflow, "set_tracking_uri")
    mocker.patch.object(
        registry.mlflow.tracking, "MlflowClient", return_value=client
    )

    assert registry.load_production_model("m", "production") is None
    assert "production" in caplog.text
