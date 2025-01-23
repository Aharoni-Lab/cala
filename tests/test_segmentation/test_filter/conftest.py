import pytest
from cala.segmentation.filter import (
    GMMFilter,
    IntensityFilter,
    DistributionFilter,
    PNRFilter,
    GLContrastFilter,
)


@pytest.fixture(
    params=[
        GMMFilter,
        IntensityFilter,
        DistributionFilter,
        PNRFilter,
        GLContrastFilter,
    ]
)
def filter_instance(request):
    """Parameterized fixture that yields instances of all filter classes."""
    return request.param()
