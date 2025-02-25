import os

import pytest
from numpy.random import RandomState

from .fixtures import params, raw_calcium_video, preprocessed_video, stabilized_video


@pytest.fixture(autouse=True)
def mock_random(monkeypatch: pytest.MonkeyPatch):
    rs = RandomState(12345)

    def stable_random(*args, **kwargs):
        return rs.random(*args, **kwargs)

    monkeypatch.setattr("numpy.random.random", stable_random)


@pytest.fixture(autouse=True)
def cleanup_numba_env():
    """Ensure NUMBA_DISABLE_JIT is reset after each test"""
    original = os.environ.get("NUMBA_DISABLE_JIT")
    yield
    if original is None:
        os.environ.pop("NUMBA_DISABLE_JIT", None)
    else:
        os.environ["NUMBA_DISABLE_JIT"] = original
