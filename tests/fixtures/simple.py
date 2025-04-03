from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr


@dataclass
class SimplyParams:
    n_components: int = 5
    height: int = 10
    width: int = 10
    n_frames: int = 20


@pytest.fixture(scope="session")
def simply_params():
    return SimplyParams()


@pytest.fixture(scope="session")
def simply_traces(simply_params, mini_coords):
    traces = xr.DataArray(
        np.zeros((simply_params.n_components, simply_params.n_frames)),
        dims=("component", "frame"),
        coords=mini_coords,
    )
    traces[0, :] = [1 for _ in range(simply_params.n_frames)]
    traces[1, :] = [i for i in range(simply_params.n_frames)]
    traces[2, :] = [
        simply_params.n_frames - 1 - i for i in range(simply_params.n_frames)
    ]
    traces[3, :] = [
        abs((simply_params.n_frames - 1) / 2 - i) for i in range(simply_params.n_frames)
    ]
    traces[4, :] = np.random.rand(simply_params.n_frames) * simply_params.n_frames

    return traces


@pytest.fixture(scope="session")
def simply_denoised(mini_footprints, simply_traces):
    return (mini_footprints @ simply_traces).transpose("frame", "height", "width")
