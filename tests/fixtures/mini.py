from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr


@dataclass
class MiniParams:
    n_components: int = 5
    height: int = 10
    width: int = 10
    n_frames: int = 5


@pytest.fixture(scope="session")
def mini_params():
    """Return default parameters for video generation."""
    return MiniParams()


@pytest.fixture(scope="session")
def mini_coords(mini_params):
    # Create sample coordinates
    return {
        "id_": ("component", [f"id{i}" for i in range(mini_params.n_components)]),
        "type_": (
            "component",
            ["background"] + ["neuron"] * (mini_params.n_components - 1),
        ),
    }


@pytest.fixture(scope="session")
def mini_footprints(mini_params, mini_coords):
    """Create sample data for testing."""
    footprints = xr.DataArray(
        np.zeros((mini_params.n_components, mini_params.height, mini_params.width)),
        dims=("component", "height", "width"),
        coords=mini_coords,
    )
    # Set up specific overlap patterns
    footprints[0, 0:5, 0:5] = 1  # Component 0
    footprints[1, 3:8, 3:8] = 1  # Component 1 (overlaps with 0)
    footprints[2, 8:10, 8:10] = 1  # Component 2 (isolated)
    footprints[3, 0:3, 8:10] = 1  # Component 3
    footprints[4, 1:4, 7:9] = 1  # Component 4 (overlaps with 3)

    return footprints


@pytest.fixture(scope="session")
def mini_traces(mini_params, mini_coords):
    traces = xr.DataArray(
        np.zeros((mini_params.n_components, mini_params.n_frames)),
        dims=("component", "frame"),
        coords=mini_coords,
    )
    traces[0, :] = [1 for _ in range(mini_params.n_frames)]
    traces[1, :] = [i for i in range(mini_params.n_frames)]
    traces[2, :] = [mini_params.n_frames - 1 - i for i in range(mini_params.n_frames)]
    traces[3, :] = [
        abs((mini_params.n_frames - 1) / 2 - i) for i in range(mini_params.n_frames)
    ]
    traces[4, :] = np.random.rand(mini_params.n_frames)

    return traces


@pytest.fixture(scope="session")
def mini_residuals(mini_params):
    residual = xr.DataArray(
        np.zeros((mini_params.n_frames, mini_params.height, mini_params.width)),
        dims=("frame", "height", "width"),
    )
    for i in range(mini_params.n_frames):
        residual[i, :, i % mini_params.width] = 3

    return residual


@pytest.fixture(scope="session")
def mini_denoised(mini_footprints, mini_traces):
    return (mini_footprints @ mini_traces).transpose("frame", "height", "width")


@pytest.fixture(scope="session")
def mini_movie(mini_denoised, mini_residuals):
    return (mini_denoised + mini_residuals).transpose("frame", "height", "width")
