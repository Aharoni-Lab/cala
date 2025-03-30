import os
from pathlib import Path

import pytest
from numpy.random import RandomState

from tests.fixtures.simulation import (
    params,
    ids,
    types,
    radii,
    positions,
    footprints,
    spikes,
    traces,
    camera_motion,
    motion_operator,
    scope_noise,
    noise,
    glow,
    hot_pixels,
    dead_pixels,
    photobleaching,
    raw_calcium_video,
    preprocessed_video,
    stabilized_video,
)

from tests.fixtures.mini import (
    mini_coords,
    mini_footprints,
    mini_traces,
    mini_residuals,
    mini_denoised,
    mini_movie,
)

from tests.viz_util import Visualizer


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


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "viz: mark test to run with visualizations (skip during CI/CD)"
    )


@pytest.fixture(scope="session")
def viz_dir():
    """Create visualization output directory within tests folder."""
    # Get the directory where tests are located
    test_dir = Path(__file__).parent
    viz_path = test_dir / "artifacts"
    viz_path.mkdir(exist_ok=True)
    return viz_path


@pytest.fixture
def visualizer(request, viz_dir):
    """Function-scoped fixture for visualization utilities."""
    # Skip if in CI or test isn't marked for viz
    if os.environ.get("CI") or not request.node.get_closest_marker("viz"):
        return None

    return Visualizer(viz_dir)
