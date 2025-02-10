from cala.batch.segmentation.filter import GMMFilter


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
