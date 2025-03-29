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
def baby_coords(p):
    # Create sample coordinates
    return {
        "id_": ("component", [f"id{i}" for i in range(p.n_components)]),
        "type_": ("component", ["background"] + ["neuron"] * (p.n_components - 1)),
    }


@pytest.fixture
def baby_footprints(p, baby_coords):
    """Create sample data for testing."""
    footprints = xr.DataArray(
        np.zeros((p.n_components, p.height, p.width)),
        dims=("component", "height", "width"),
        coords=baby_coords,
    )
    footprints[0, 0:2, 0:2] = 1
    footprints[1, 1:4, 1:4] = 3
    footprints[2, 3:5, 3:5] = 2

    return footprints


@pytest.fixture
def baby_traces(p, baby_coords):
    traces = xr.DataArray(
        np.zeros((p.n_components, p.n_frames)),
        dims=("component", "frame"),
        coords=baby_coords,
    )
    traces[0, :] = [1 for _ in range(p.n_frames)]
    traces[1, :] = [i for i in range(p.n_frames)]
    traces[2, :] = [p.n_frames - i for i in range(p.n_frames)]

    return traces


@pytest.fixture
def baby_residuals(p):
    residual = xr.DataArray(
        np.zeros((p.n_frames, p.height, p.width)), dims=("frame", "height", "width")
    )
    for i in range(p.n_frames):
        residual[i, :, i % p.width] = 3

    return residual


@pytest.fixture
def baby_denoised(baby_footprints, baby_traces):
    return baby_footprints @ baby_traces


@pytest.fixture
def baby_movie(baby_denoised, baby_residuals):
    return baby_denoised + baby_residuals
