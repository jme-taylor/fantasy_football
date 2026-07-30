"""Shared fixtures for unit tests.

Unit tests must run hermetically, without depending on developer secrets or a
local ``.env``. Several extractors build a ``GitHubAPIClient`` during
construction, which requires ``GITHUB_API_KEY``; locally a ``.env`` supplies it,
but CI has none. Provide a dummy key for every unit test so construction never
depends on the ambient environment. No real requests are made: unit tests mock
the network, and live-network tests are marked ``integration`` and deselected.
"""

from collections.abc import Iterator
from pathlib import Path

import duckdb
import pytest

from fantasy_football.storage.database import get_connection


@pytest.fixture(autouse=True)
def _dummy_github_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a dummy ``GITHUB_API_KEY`` so client construction never needs a real one."""
    monkeypatch.setenv("GITHUB_API_KEY", "test-token")


@pytest.fixture
def db(tmp_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    """Open an on-disk connection with every table created, then close it.

    Parameters
    ----------
    tmp_path : Path
        Pytest's per-test temporary directory.

    Yields
    ------
    duckdb.DuckDBPyConnection
        An open connection with every stored table created.
    """
    connection = get_connection(tmp_path / "test.duckdb")
    yield connection
    connection.close()
