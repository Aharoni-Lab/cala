from dataclasses import dataclass
from enum import Enum

import numpy as np
import pytest
import sparse
import xarray as xr

from cala.streaming.core import Component


@dataclass
class MiniParams:
    n_components: int = 5
    height: int = 10
    width: int = 10
    n_frames: int = 5


@pytest.fixture(scope="session")
def mini_params() -> MiniParams:
    """Return default parameters for video generation."""
    return MiniParams()


@pytest.fixture(scope="session")
def mini_coords(mini_params: MiniParams) -> dict[str, tuple[str, list[str | Enum]]]:
    # Create sample coordinates
    return {
        "id_": ("component", [f"id{i}" for i in range(mini_params.n_components)]),
        "type_": (
            "component",
            [Component.BACKGROUND] + [Component.NEURON] * (mini_params.n_components - 1),
        ),
    }


@pytest.fixture(scope="session")
def mini_footprints(
    mini_params: MiniParams, mini_coords: dict[str, tuple[str, str | Enum]]
) -> xr.DataArray:
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
def mini_traces(
    mini_params: MiniParams, mini_coords: dict[str, tuple[str, str | Enum]]
) -> xr.DataArray:
    traces = xr.DataArray(
        np.zeros((mini_params.n_components, mini_params.n_frames)),
        dims=("component", "frame"),
        coords=mini_coords,
    )
    traces[0, :] = [1 for _ in range(mini_params.n_frames)]
    traces[1, :] = [i for i in range(mini_params.n_frames)]
    traces[2, :] = [mini_params.n_frames - 1 - i for i in range(mini_params.n_frames)]
    traces[3, :] = [abs((mini_params.n_frames - 1) / 2 - i) for i in range(mini_params.n_frames)]
    traces[4, :] = np.random.rand(mini_params.n_frames)

    return traces


@pytest.fixture(scope="session")
def mini_residuals(mini_params: MiniParams) -> xr.DataArray:
    residual = xr.DataArray(
        np.zeros((mini_params.n_frames, mini_params.height, mini_params.width)),
        dims=("frame", "height", "width"),
    )
    for i in range(mini_params.n_frames):
        residual[i, :, i % mini_params.width] = 3

    return residual


@pytest.fixture(scope="session")
def mini_pixel_stats(
    mini_params: MiniParams, mini_denoised: xr.DataArray, mini_traces: xr.DataArray
) -> xr.DataArray:
    # Get current timestep
    t_prime = mini_params.n_frames

    # Reshape frames to pixels x time
    Y = mini_denoised.stack({"pixels": ("width", "height")})

    # Get temporal components C
    C = mini_traces  # components x time

    # Compute W = Y[:, 1:t']C^T/t'
    W = Y @ C.T / t_prime

    # Create xarray DataArray with proper dimensions and coordinates
    return W.unstack("pixels")


@pytest.fixture(scope="session")
def mini_component_stats(mini_params: MiniParams, mini_traces: xr.DataArray) -> xr.DataArray:
    t_prime = mini_params.n_frames

    # Get temporal components C
    C = mini_traces  # components x time

    # Compute M = C * C.T / t'
    M = C @ C.rename({"component": "component'"}) / t_prime

    return M.assign_coords(C.coords)


@pytest.fixture(scope="session")
def mini_overlaps(mini_footprints: xr.DataArray) -> xr.DataArray:
    data = (mini_footprints.dot(mini_footprints.rename({"component": "component'"})) > 0).astype(
        int
    )

    data.values = sparse.COO(data.values)
    return data.assign_coords(
        {
            "id_": (
                "component",
                mini_footprints.coords["id_"].values,
            ),
            "type_": (
                "component",
                mini_footprints.coords["type_"].values,
            ),
        }
    )


@pytest.fixture(scope="session")
def mini_denoised(mini_footprints: xr.DataArray, mini_traces: xr.DataArray) -> xr.DataArray:
    return (mini_footprints @ mini_traces).transpose("frame", "height", "width")


@pytest.fixture(scope="session")
def mini_movie(mini_denoised: xr.DataArray, mini_residuals: xr.DataArray) -> xr.DataArray:
    return (mini_denoised + mini_residuals).transpose("frame", "height", "width")
