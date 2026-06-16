"""Shared fixtures for unit tests.

Unit tests must run hermetically, without depending on developer secrets or a
local ``.env``. Several extractors build a ``GitHubAPIClient`` during
construction, which requires ``GITHUB_API_KEY``; locally a ``.env`` supplies it,
but CI has none. Provide a dummy key for every unit test so construction never
depends on the ambient environment. No real requests are made: unit tests mock
the network, and live-network tests are marked ``integration`` and deselected.
"""

import pytest


@pytest.fixture(autouse=True)
def _dummy_github_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provide a dummy ``GITHUB_API_KEY`` so client construction never needs a real one."""
    monkeypatch.setenv("GITHUB_API_KEY", "test-token")
