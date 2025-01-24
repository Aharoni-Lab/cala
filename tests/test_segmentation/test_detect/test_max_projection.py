import pytest

from cala.segmentation.detect import MaxProjection


def test_max_projection_basic(stabilized_video):
    """Test basic functionality of MaxProjection detector."""
    video, ground_truth, metadata = stabilized_video

    # Create detector with default parameters
    detector = MaxProjection(core_axes=["height", "width"], iter_axis="frames")

    # Detect seeds
    seeds = detector.fit_transform(video)

    # Basic checks
    assert len(seeds) > 0, "No seeds detected"
    assert all(
        col in seeds.columns for col in ["height", "width"]
    ), "Missing coordinate columns"
    assert len(seeds) >= 0.8 * len(ground_truth), "Too few seeds detected"
    assert len(seeds) <= 1.2 * len(ground_truth), "Too many seeds detected"


def test_detection_with_different_cell_sizes(stabilized_video):
    """Test detection with varying cell sizes."""
    video, ground_truth, _ = stabilized_video

    # Test with different local_max_radius values
    radii = [5, 10, 15]
    results = []

    for radius in radii:
        detector = MaxProjection(
            core_axes=["height", "width"], iter_axis="frames", local_max_radius=radius
        )
        seeds = detector.fit_transform(video)
        results.append(len(seeds))

    # Expect medium radius to perform best
    assert results[0] >= results[1], "Medium radius should detect fewer than small"
    assert results[1] >= results[2], "Medium radius should detect more than large"


@pytest.mark.parametrize("intensity_threshold", [1, 2, 3])
def test_intensity_threshold_effect(stabilized_video, intensity_threshold):
    """Test effect of intensity threshold on detection."""
    video, ground_truth, _ = stabilized_video

    detector = MaxProjection(
        core_axes=["height", "width"],
        iter_axis="frames",
        intensity_threshold=intensity_threshold,
    )

    seeds = detector.fit_transform(video)

    # Higher threshold should detect fewer seeds
    if intensity_threshold > 1:
        prev_detector = MaxProjection(
            core_axes=["height", "width"],
            iter_axis="frames",
            intensity_threshold=intensity_threshold - 1,
        )
        prev_seeds = prev_detector.fit_transform(video)
        assert len(seeds) <= len(
            prev_seeds
        ), "Higher threshold should detect fewer seeds"
