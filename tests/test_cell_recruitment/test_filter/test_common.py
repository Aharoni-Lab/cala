import pytest
import numpy as np
import pandas as pd
import xarray as xr

from cala.cell_recruitment.filter.base import BaseFilter


def test_filter_initialization(filter_instance):
    """Test that all filters initialize properly."""
    assert isinstance(filter_instance, BaseFilter)


def test_filter_interface(filter_instance):
    """Test that all filters implement the required interface."""
    assert hasattr(filter_instance, "fit")
    assert hasattr(filter_instance, "transform")
    assert hasattr(filter_instance, "fit_transform")
    assert hasattr(filter_instance, "core_axes")
    assert hasattr(filter_instance, "iter_axis")


def test_filter_accuracy(filter_instance, stabilized_video, noisy_seeds):
    """Test accuracy metrics for all filters with different synthetic data."""
    video, _, _ = stabilized_video

    max_pixel_value = float(video.max())

    for _, row in noisy_seeds.iterrows():
        if row["is_real"] is False:
            video[:, int(row["height"]), int(row["width"])] = (
                np.random.normal(0, 1, video.shape[0])
                / max_pixel_value  # an arbirary division factor to make the noise smaller than the signal
            )

    # Apply the filter
    filtered_seeds = filter_instance.fit_transform(video, noisy_seeds)

    # Get the mask column name for this filter
    mask_col = next(col for col in filtered_seeds.columns if col.startswith("mask_"))

    # Calculate accuracy metrics
    true_positives = filtered_seeds[
        filtered_seeds["is_real"] & filtered_seeds[mask_col]
    ]
    false_positives = filtered_seeds[
        ~filtered_seeds["is_real"] & filtered_seeds[mask_col]
    ]
    false_negatives = filtered_seeds[
        ~filtered_seeds["is_real"] & ~filtered_seeds[mask_col]
    ]

    # The filter should keep most real cells
    recall = len(true_positives) / len(filtered_seeds[filtered_seeds["is_real"]])

    # The filter should remove most false positives
    false_positive_rate = len(false_positives) / len(
        filtered_seeds[~filtered_seeds["is_real"]]
    )

    # f1 metric
    f1_score = len(true_positives) / (
        len(true_positives) + 0.5 * (len(false_positives) + len(false_negatives))
    )
    assert (
        f1_score >= 0.75
    ), f"{filter_instance.__class__.__name__}'s f1 score is too low."


def test_filter_transform_without_fit(filter_instance):
    """Test error handling when transform is called before fit (if required)."""
    dummy_data = xr.DataArray(
        data=np.random.rand(10, 10, 10), dims=["width", "height", "frames"]
    )
    dummy_seeds = pd.DataFrame({"height": [5], "width": [5]})

    if filter_instance.reusing_fit:
        with pytest.raises(ValueError, match="The filter has not been fitted."):
            filter_instance.transform(dummy_data, dummy_seeds)
    else:
        # Should work without fitting for stateless filters
        result = filter_instance.transform(dummy_data, dummy_seeds)
        mask_col = next(col for col in result.columns if col.startswith("mask_"))
        assert mask_col in result.columns


def test_filter_data_validation(filter_instance, stabilized_video, noisy_seeds):
    """Test that filters properly validate their input data."""
    video, _, _ = stabilized_video

    # Test with missing required columns
    bad_seeds = noisy_seeds.drop(columns=[filter_instance.core_axes[0]])
    with pytest.raises(KeyError):
        filter_instance.fit_transform(video, bad_seeds)

    # Test with wrong dimensionality
    bad_video = video.isel({filter_instance.iter_axis: 0})
    with pytest.raises(ValueError):
        filter_instance.fit_transform(bad_video, noisy_seeds)
