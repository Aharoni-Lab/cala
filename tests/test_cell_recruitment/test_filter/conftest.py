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
