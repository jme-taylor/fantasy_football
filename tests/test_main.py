import polars as pl
from pytest_mock import MockerFixture

import main as main_module
from fantasy_football.seasons import DataSource


def test_update_current_season_routes_to_fci_for_current(
    mocker: MockerFixture,
) -> None:
    """Current/future seasons are refreshed via the FCI extractor."""
    mocker.patch.object(
        main_module, "source_for_season", return_value=DataSource.FCI
    )
    fci = mocker.patch.object(main_module, "FciExtractor")
    main_module.update_current_season("2025-26")
    fci.return_value.build_current_season_merged_gw.assert_called_once_with(
        "2025-26"
    )


def _patch_pipeline(mocker: MockerFixture) -> None:
    """Patch out every side-effecting pipeline step main() calls."""
    for name in (
        "configure_logging",
        "update_current_season",
        "create_rolling_points_data",
        "build_fixtures_enriched",
        "build_team_elo",
        "predict_points",
    ):
        mocker.patch.object(main_module, name)


def test_main_skips_optimisation_when_predictions_empty(
    mocker: MockerFixture,
) -> None:
    """An empty predictions file skips optimisation instead of crashing."""
    _patch_pipeline(mocker)
    mocker.patch.object(
        main_module.pl,
        "read_csv",
        return_value=pl.DataFrame(schema={"gw": pl.Int64}),
    )
    optimise = mocker.patch.object(main_module, "optimise_plan")

    main_module.main()

    optimise.assert_not_called()


def test_main_optimises_when_predictions_present(
    mocker: MockerFixture,
) -> None:
    """A non-empty predictions file drives optimisation from the min gw."""
    _patch_pipeline(mocker)
    mocker.patch.object(
        main_module.pl,
        "read_csv",
        return_value=pl.DataFrame({"gw": [7, 8, 9]}),
    )
    optimise = mocker.patch.object(main_module, "optimise_plan")

    main_module.main()

    optimise.assert_called_once_with(main_module.CURRENT_SEASON, 7)
