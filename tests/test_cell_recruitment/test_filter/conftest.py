import pytest
from cala.cell_recruitment.filter import (
    GMMFilter,
    IntensityFilter,
    DistributionFilter,
    PNRFilter,
)
import numpy as np


@pytest.fixture(
    params=[
        GMMFilter,
        IntensityFilter,
        DistributionFilter,
        PNRFilter,
    ]
)
def filter_instance(request):
    """Parameterized fixture that yields instances of all filter classes."""
    return request.param()


@pytest.fixture
def synthetic_seeds(noisy_seeds, stabilized_video):
    """Generate different types of synthetic seeds based on the data type."""
    video, _, _ = stabilized_video
    seeds = noisy_seeds.copy()

    # Add temporal pattern information (normal-like distributions)
    traces = []
    max_pixel_value = float(video.max())

    for _, row in seeds.iterrows():
        if row["is_real"]:
            trace = video.isel(
                height=int(row["height"]), width=int(row["width"])
            ).values
        else:
            trace = np.random.rand(video.shape[0]) * max_pixel_value
        traces.append(trace)

    return traces
