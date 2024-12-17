import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cala.cell_recruitment.filter import GMMFilter


def test_gmm_filter_initialization():
    """Test that GMMFilter initializes with default parameters."""
    filter = GMMFilter()
    assert filter.quantile_floor == 0.1
    assert filter.quantile_ceil == 0.99
    assert filter.num_components == 2
    assert filter.num_valid_components == 1
    assert filter.mean_mask is True


def test_gmm_filter_invalid_quantiles():
    """Test that GMMFilter raises error for invalid quantiles."""
    with pytest.raises(
        ValueError, match="quantile_floor must be smaller than quantile_ceil"
    ):
        GMMFilter(quantile_floor=0.9, quantile_ceil=0.1)

    with pytest.raises(ValueError, match="quantiles must be between 0 and 1"):
        GMMFilter(quantile_floor=-0.1)


def test_gmm_filter_fit_transform(stabilized_video, noisy_seeds):
    """Test that GMMFilter correctly identifies real cells."""
    video, _, _ = stabilized_video

    # Initialize and fit the filter
    filter = GMMFilter()
    filtered_seeds = filter.fit_transform(video, noisy_seeds)

    # Check that the filter adds a mask column
    assert "mask_gmm" in filtered_seeds.columns

    # Calculate accuracy metrics
    true_positives = filtered_seeds[
        filtered_seeds["is_real"] & filtered_seeds["mask_gmm"]
    ]
    false_positives = filtered_seeds[
        ~filtered_seeds["is_real"] & filtered_seeds["mask_gmm"]
    ]

    # The filter should keep most real cells
    recall = len(true_positives) / len(filtered_seeds[filtered_seeds["is_real"]])
    assert recall > 0.8, "Filter removed too many real cells"

    # The filter should remove most false positives
    false_positive_rate = len(false_positives) / len(
        filtered_seeds[~filtered_seeds["is_real"]]
    )
    assert false_positive_rate < 0.3, "Filter kept too many false positives"


def test_gmm_filter_components(stabilized_video, noisy_seeds):
    """Test that GMMFilter correctly uses multiple components."""
    video, _, _ = stabilized_video

    # Test with different numbers of components
    filter_1 = GMMFilter(num_components=2, num_valid_components=1)
    filter_2 = GMMFilter(num_components=3, num_valid_components=2)

    results_1 = filter_1.fit_transform(video, noisy_seeds)
    results_2 = filter_2.fit_transform(video, noisy_seeds)

    # More components with more valid components should keep more seeds
    assert results_2["mask_gmm"].sum() >= results_1["mask_gmm"].sum()


def test_gmm_filter_without_fit():
    """Test that GMMFilter raises error when transform is called before fit."""
    filter = GMMFilter()
    with pytest.raises(ValueError, match="Transformer has not been fitted yet"):
        filter.transform(xr.DataArray(np.random.rand(10, 10, 10)), pd.DataFrame())
