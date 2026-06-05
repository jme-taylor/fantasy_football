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
