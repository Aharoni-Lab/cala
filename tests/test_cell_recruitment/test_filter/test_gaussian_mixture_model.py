import numpy as np
import pandas as pd
import pytest
import xarray as xr
from sklearn.mixture import GaussianMixture

from cala.cell_recruitment.filter import GMMFilter


@pytest.fixture
def generate_test_data():
    """Fixture to generate synthetic test data for varr and seeds."""
    time_points = 100
    num_spatial_points = 50
    np.random.seed(42)

    # Simulate fluorescence data with two different distributions
    # Valid seeds with higher fluctuations
    valid_seed_indices = np.random.choice(num_spatial_points, size=25, replace=False)
    valid_fluctuations = np.random.normal(loc=5, scale=2, size=(25, time_points))

    # Invalid seeds with lower fluctuations
    invalid_seed_indices = np.setdiff1d(
        np.arange(num_spatial_points), valid_seed_indices
    )
    invalid_fluctuations = np.random.normal(loc=2, scale=0.5, size=(25, time_points))

    all_fluctuations = np.zeros((num_spatial_points, time_points))
    all_fluctuations[valid_seed_indices, :] = valid_fluctuations
    all_fluctuations[invalid_seed_indices, :] = invalid_fluctuations

    heights = np.arange(num_spatial_points)
    widths = np.arange(num_spatial_points)
    spatial_index = pd.MultiIndex.from_arrays(
        [heights, widths], names=("height", "width")
    )

    video = xr.DataArray(
        all_fluctuations,
        dims=["spatial", "frames"],
        coords={
            "spatial": spatial_index,
            "frames": np.arange(time_points),
        },
    )

    seeds = pd.DataFrame(
        {
            "height": np.arange(num_spatial_points),
            "width": np.arange(num_spatial_points),
        }
    )

    return video, seeds, valid_seed_indices


def test_gmm_refine_basic(generate_test_data):
    """Test the basic functionality of gmm_refine."""
    video, seeds, valid_seed_indices = generate_test_data

    gmm_filter = GMMFilter()
    seeds_result = gmm_filter.fit_transform(X=video, y=seeds)

    # Check that the mask_gmm column is added
    assert (
        "mask_gmm" in seeds_result.columns
    ), "mask_gmm column not added to seeds DataFrame"

    # Verify that the correct seeds are marked as valid
    mask_gmm = seeds_result["mask_gmm"].values
    # We expect the valid seeds to be marked as True
    expected_mask = np.zeros(len(seeds), dtype=bool)
    expected_mask[valid_seed_indices] = True

    assert np.array_equal(
        mask_gmm, expected_mask
    ), "Valid seeds not correctly identified"


def test_gmm_refine_mean_mask_false(generate_test_data):
    """Test gmm_refine with mean_mask set to False."""
    video, seeds, valid_seed_indices = generate_test_data

    # Run with mean_mask=False
    gmm_filter = GMMFilter()
    seeds_result = gmm_filter.fit_transform(X=video, y=seeds)

    # Check that the mask_gmm column is added
    assert (
        "mask_gmm" in seeds_result.columns
    ), "mask_gmm column not added to seeds DataFrame"

    # Verify that the correct number of seeds are marked as valid
    # Without mean_mask, some invalid seeds may be marked as valid if distributions overlap
    num_valid_seeds = seeds_result["mask_gmm"].sum()
    assert num_valid_seeds >= len(
        valid_seed_indices
    ), "Number of valid seeds is less than expected"


def test_gmm_refine_different_q(generate_test_data):
    """Test gmm_refine with different percentile values."""
    video, seeds, valid_seed_indices = generate_test_data

    # Use different percentiles for peak-to-peak calculation
    gmm_filter = GMMFilter(quantile_floor=0.25, quantile_ceil=0.75)
    seeds_result = gmm_filter.fit_transform(X=video, y=seeds)
    # Check that the mask_gmm column is added
    assert (
        "mask_gmm" in seeds_result.columns
    ), "mask_gmm column not added to seeds DataFrame"

    # Verify that the correct seeds are marked as valid
    mask_gmm = seeds_result["mask_gmm"].values
    expected_mask = np.zeros(len(seeds), dtype=bool)
    expected_mask[valid_seed_indices] = True

    assert np.array_equal(
        mask_gmm, expected_mask
    ), "Valid seeds not correctly identified with different q values"


def test_gmm_refine_return_types(generate_test_data):
    """Test that gmm_refine returns the correct types."""
    video, seeds, _ = generate_test_data

    gmm_filter = GMMFilter()
    seeds_result = gmm_filter.fit_transform(X=video, y=seeds)

    assert isinstance(
        seeds_result, pd.DataFrame
    ), "seeds_result should be a pandas DataFrame"
