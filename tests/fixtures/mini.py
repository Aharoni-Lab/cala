from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr


@dataclass
class SanityParams:
    n_components: int = 3
    height: int = 5
    width: int = 5
    n_frames: int = 15


@pytest.fixture
def p():
    """Return default parameters for video generation."""
    return SanityParams()


@pytest.fixture
def mini_coords(p):
    # Create sample coordinates
    return {
        "id_": ("component", [f"id{i}" for i in range(p.n_components)]),
        "type_": ("component", ["background"] + ["neuron"] * (p.n_components - 1)),
    }


@pytest.fixture
def mini_footprints(p, mini_coords):
    """Create sample data for testing."""
    footprints = xr.DataArray(
        np.zeros((p.n_components, p.height, p.width)),
        dims=("component", "height", "width"),
        coords=mini_coords,
    )
    footprints[0, 0:2, 0:2] = 1
    footprints[1, 1:4, 1:4] = 3
    footprints[2, 3:5, 3:5] = 2

    return footprints


@pytest.fixture
def mini_traces(p, mini_coords):
    traces = xr.DataArray(
        np.zeros((p.n_components, p.n_frames)),
        dims=("component", "frame"),
        coords=mini_coords,
    )
    traces[0, :] = [1 for _ in range(p.n_frames)]
    traces[1, :] = [i for i in range(p.n_frames)]
    traces[2, :] = [p.n_frames - i for i in range(p.n_frames)]

    return traces


@pytest.fixture
def mini_residuals(p):
    residual = xr.DataArray(
        np.zeros((p.n_frames, p.height, p.width)), dims=("frame", "height", "width")
    )
    for i in range(p.n_frames):
        residual[i, :, i % p.width] = 3

    return residual


@pytest.fixture
def mini_denoised(mini_footprints, mini_traces):
    return mini_footprints @ mini_traces


@pytest.fixture
def mini_movie(mini_denoised, mini_residuals):
    return mini_denoised + mini_residuals
