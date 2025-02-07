from cala.batch.segmentation.filter import IntensityFilter


def test_intensity_filter_threshold_scaling(stabilized_video, noisy_seeds):
    """Test that intensity threshold scales with seed_intensity_factor."""
    video, _, _ = stabilized_video

    # Test with different scaling factors
    filter_1 = IntensityFilter(seed_intensity_factor=1)
    filter_2 = IntensityFilter(seed_intensity_factor=2)

    # Fit both filters
    filter_1.fit(video, noisy_seeds)
    filter_2.fit(video, noisy_seeds)

    # Higher factor should lead to higher threshold
    assert (
        filter_2.intensity_threshold_ >= filter_1.intensity_threshold_
    ), "Higher intensity factor should result in higher threshold"
