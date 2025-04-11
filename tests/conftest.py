import inspect
import os
from collections.abc import Generator
from pathlib import Path
from typing import Any

import pytest
from numpy.random import RandomState

from cala.viz_util import Visualizer
from tests.fixtures.mini import (
    mini_comp_coords,
    mini_component_stats,
    mini_denoised,
    mini_footprints,
    mini_frame_coords,
    mini_movie,
    mini_overlaps,
    mini_params,
    mini_pixel_stats,
    mini_residuals,
    mini_traces,
)
from tests.fixtures.simple import simply_coords, simply_denoised, simply_params, simply_traces
from tests.fixtures.simulation import (
    camera_motion,
    dead_pixels,
    footprints,
    frame_coords,
    glow,
    hot_pixels,
    ids,
    motion_operator,
    noise,
    params,
    photobleaching,
    positions,
    preprocessed_video,
    radii,
    raw_calcium_video,
    scope_noise,
    spikes,
    stabilized_video,
    traces,
    types,
)


@pytest.fixture(autouse=True)
def mock_random(monkeypatch: pytest.MonkeyPatch) -> None:
    rs = RandomState(12345)

    def stable_random() -> Any:
        return rs.random()

    monkeypatch.setattr("numpy.random.random", stable_random)


@pytest.fixture(autouse=True)
def cleanup_numba_env() -> Generator:
    """Ensure NUMBA_DISABLE_JIT is reset after each test"""
    original = os.environ.get("NUMBA_DISABLE_JIT")
    yield
    if original is None:
        os.environ.pop("NUMBA_DISABLE_JIT", None)
    else:
        os.environ["NUMBA_DISABLE_JIT"] = original


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers", "viz: mark test to run with visualizations (skip during CI/CD)"
    )


@pytest.fixture(scope="session")
def viz_dir() -> Path:
    """Create visualization output directory within tests folder."""
    # Get the directory where tests are located
    test_dir = Path(__file__).parent
    viz_path = test_dir / "artifacts"
    viz_path.mkdir(exist_ok=True)
    return viz_path


class NullVisualizer:
    def __init__(self, *args: Any, **kwargs: Any):
        pass


@pytest.fixture
def visualizer(request: pytest.FixtureRequest, viz_dir: Path) -> NullVisualizer | Visualizer:
    """Function-scoped fixture for visualization utilities."""
    # Skip if in CI or test isn't marked for viz
    if os.environ.get("CI") or not request.node.get_closest_marker("viz"):

        def create_null_visualizer() -> type:
            # Get all methods from the real Visualizer class
            for name, _ in inspect.getmembers(Visualizer, predicate=inspect.isfunction):
                # Skip magic methods
                if not name.startswith("_"):
                    # Create a no-op method with the same name
                    setattr(NullVisualizer, name, lambda self, *args, **kwargs: None)

            return NullVisualizer

        return create_null_visualizer()(viz_dir)

    return Visualizer(viz_dir)
