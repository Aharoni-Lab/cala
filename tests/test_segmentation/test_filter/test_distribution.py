import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cala.segmentation.filter import DistributionFilter


def test_distribution_filter_kernel():
    """Test the KS test kernel function directly."""
    # Test with perfectly normal distribution
    normal_data = np.random.normal(0, 1, 1000)
    n_components_normal = DistributionFilter.min_ic_components(normal_data)
    assert n_components_normal == 1, "1 component should be ideal for normal data"

    # Test with constant data
    constant_data = np.ones(1000)
    n_components_constant = DistributionFilter.min_ic_components(constant_data)
    assert n_components_constant == 1, "1 component should be ideal for constant data"

    # Test with clearly non-normal distribution (e.g., uniform)
    uniform_data = np.random.uniform(0, 1, 1000)
    n_components_uniform = DistributionFilter.min_ic_components(uniform_data)
    assert (
        n_components_uniform > n_components_constant
    ), "Uniform data should be less normal"
