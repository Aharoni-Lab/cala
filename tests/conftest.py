from typing import Any

import pytest
from numpy.random import RandomState


@pytest.fixture(autouse=True)
def mock_random(monkeypatch: pytest.MonkeyPatch) -> None:
    rs = RandomState(12345)

    def stable_random() -> Any:
        return rs.random()

    monkeypatch.setattr("numpy.random.random", stable_random)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "viz: mark test to run with visualizations (skip during CI/CD)"
    )
