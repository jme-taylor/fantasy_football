from unittest.mock import MagicMock, patch

import main as main_module


def test_load_player_match_data_calls_both_sources() -> None:
    """The player_match loader runs the historic and current-season paths."""
    connection = MagicMock()
    extractor = MagicMock()
    current_loader = MagicMock()

    main_module.load_player_match_data(
        "2025-26",
        connection,
        extractor=extractor,
        current_loader=current_loader,
    )

    extractor.load_immutable_player_match_seasons.assert_called_once_with(
        connection, "2025-26"
    )
    current_loader.assert_called_once_with("2025-26", connection)


def test_train_minutes_model_invokes_runner() -> None:
    """The helper runs the minutes-model entrypoint once."""
    with patch.object(main_module, "run_minutes_model") as runner:
        main_module._train_minutes_model()

    runner.assert_called_once_with()


def test_train_minutes_model_is_non_fatal() -> None:
    """A failure in the minutes model is swallowed, not raised."""
    with patch.object(
        main_module, "run_minutes_model", side_effect=RuntimeError("boom")
    ):
        # Must not raise.
        main_module._train_minutes_model()
