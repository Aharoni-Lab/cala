import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cala.cell_recruitment.filter import IntensityFilter


@pytest.fixture
def dark_noisy_seeds(noisy_seeds, stabilized_video):
    """Add intensity information to noisy seeds for intensity filter testing."""
    video, _, _ = stabilized_video
    seeds = noisy_seeds.copy()

    # Get max projection and compute intensities for each seed
    max_proj = video.max("frames")

    # Add intensity values for each seed
    intensities = []
    for _, row in seeds.iterrows():
        intensity = float(
            max_proj.isel(height=int(row["height"]), width=int(row["width"]))
        )
        intensities.append(intensity)

    seeds["intensity"] = intensities

    # Ensure false positives are primarily in darker regions
    median_intensity = float(max_proj.median())
    seeds.loc[
        ~seeds["is_real"] & (seeds["intensity"] > median_intensity), "is_real"
    ] = True

    return seeds


def test_intensity_filter_initialization():
    """Test that IntensityFilter initializes with default parameters."""
    filter = IntensityFilter()
    assert filter.seed_intensity_factor == 2
    assert filter.max_brightness_projection_ is None
    assert filter.intensity_threshold_ is None


def test_intensity_filter_fit_transform(stabilized_video, dark_noisy_seeds):
    """Test that IntensityFilter correctly identifies real cells."""
    video, _, _ = stabilized_video

    # Initialize and fit the filter
    filter = IntensityFilter()
    filtered_seeds = filter.fit_transform(video, dark_noisy_seeds)

    # Check that the filter adds a mask column
    assert "mask_int" in filtered_seeds.columns

    # Calculate accuracy metrics
    true_positives = filtered_seeds[
        filtered_seeds["is_real"] & filtered_seeds["mask_int"]
    ]
    false_positives = filtered_seeds[
        ~filtered_seeds["is_real"] & filtered_seeds["mask_int"]
    ]

    # The filter should keep most real cells
    recall = len(true_positives) / len(filtered_seeds[filtered_seeds["is_real"]])
    assert recall > 0.8, "Filter removed too many real cells"

    # The filter should remove most false positives
    false_positive_rate = len(false_positives) / len(
        filtered_seeds[~filtered_seeds["is_real"]]
    )
    assert false_positive_rate < 0.3, "Filter kept too many false positives"


def test_intensity_filter_without_fit():
    """Test that IntensityFilter raises error when transform is called before fit."""
    filter = IntensityFilter()
    with pytest.raises(ValueError, match="Transformer has not been fitted yet"):
        filter.transform(xr.DataArray(np.random.rand(10, 10, 10)), pd.DataFrame())


def test_intensity_filter_threshold_scaling(stabilized_video, dark_noisy_seeds):
    """Test that intensity threshold scales with seed_intensity_factor."""
    video, _, _ = stabilized_video

    # Test with different scaling factors
    filter_1 = IntensityFilter(seed_intensity_factor=1)
    filter_2 = IntensityFilter(seed_intensity_factor=2)

    # Fit both filters
    filter_1.fit(video)
    filter_2.fit(video)

    # Higher factor should lead to higher threshold
    assert (
        filter_2.intensity_threshold_ >= filter_1.intensity_threshold_
    ), "Higher intensity factor should result in higher threshold"


def test_intensity_filter_one_shot(stabilized_video, dark_noisy_seeds):
    """Test one_shot parameter behavior."""
    video, _, _ = stabilized_video

    # Test with one_shot=True (default)
    filter_one_shot = IntensityFilter(one_shot=True)
    filter_one_shot.fit(video)
    initial_projection = filter_one_shot.max_brightness_projection_.copy()

    # Transform with different data
    different_video = video * 2  # Scale the video
    filter_one_shot.transform(different_video, dark_noisy_seeds)

    # Max projection should not change with one_shot=True
    xr.testing.assert_equal(
        filter_one_shot.max_brightness_projection_, initial_projection
    )

    # Test with one_shot=False
    filter_multi = IntensityFilter(one_shot=False)
    filter_multi.fit(video)
    initial_projection = filter_multi.max_brightness_projection_.copy()

    # Transform with different data
    filter_multi.transform(different_video, dark_noisy_seeds)

    # Max projection should change with one_shot=False
    with pytest.raises(AssertionError):
        xr.testing.assert_equal(
            filter_multi.max_brightness_projection_, initial_projection
        )
