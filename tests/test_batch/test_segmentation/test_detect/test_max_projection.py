from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import xarray as xr
from matplotlib.figure import Figure

from cala.batch.segmentation.detect import MaxProjection
from tests.test_batch.test_segmentation.test_detect.conftest import visualize_detection


def test_max_projection_basic(stabilized_video: xr.DataArray, params: Any) -> None:
    """Test basic functionality of MaxProjection detector."""
    video = stabilized_video

    # Create detector with default parameters
    detector = MaxProjection(core_axes=["height", "width"], iter_axis="frame")

    # Detect seeds
    seeds = detector.fit_transform(video)

    # Basic checks
    assert len(seeds) > 0, "No seeds detected"
    assert all(col in seeds.columns for col in ["height", "width"]), "Missing coordinate columns"
    assert len(seeds) >= 0.5 * params.num_neurons, "Too few seeds detected"
    assert len(seeds) <= 1.2 * params.num_neurons, "Too many seeds detected"


@pytest.fixture
def artifact_params(params: Any) -> Any:
    """Parameters for testing with strong artifacts."""
    params.photobleaching_decay = 0.8  # Strong photobleaching
    params.dead_pixel_fraction = 0.01  # More dead pixels
    params.hot_pixel_fraction = 0.005  # More hot pixels
    return params


@pytest.fixture
def size_variation_params(params: Any) -> Any:
    """Parameters for testing with varying cell sizes."""
    params.neuron_size_range = (5, 20)  # Wider range
    params.neuron_shape_irregularity = 0.4  # More irregular
    return params


def test_detection_with_different_cell_sizes(
    stabilized_video: xr.DataArray, size_variation_params: Any
) -> None:
    """Test detection with varying cell sizes."""
    video = stabilized_video

    # Test with different local_max_radius values
    radii = [5, 10, 15]
    results = []

    for radius in radii:
        detector = MaxProjection(
            core_axes=["height", "width"], iter_axis="frame", local_max_radius=radius
        )
        seeds = detector.fit_transform(video)
        results.append(len(seeds))

    # Expect medium radius to perform best
    assert results[0] >= results[1], "Medium radius should detect fewer than small"
    assert results[1] >= results[2], "Medium radius should detect more than large"


@pytest.mark.parametrize("intensity_threshold", [1, 2, 3])
def test_intensity_threshold_effect(
    stabilized_video: xr.DataArray, intensity_threshold: int
) -> None:
    """Test effect of intensity threshold on detection."""
    video = stabilized_video

    detector = MaxProjection(
        core_axes=["height", "width"],
        iter_axis="frame",
        intensity_threshold=intensity_threshold,
    )

    seeds = detector.fit_transform(video)

    # Higher threshold should detect fewer seeds
    if intensity_threshold > 1:
        prev_detector = MaxProjection(
            core_axes=["height", "width"],
            iter_axis="frame",
            intensity_threshold=intensity_threshold - 1,
        )
        prev_seeds = prev_detector.fit_transform(video)
        assert len(seeds) <= len(prev_seeds), "Higher threshold should detect fewer seeds"


def test_visualization(stabilized_video: xr.DataArray, positions: pd.DataFrame) -> None:
    """Test visualization of detection results."""
    video = stabilized_video

    detector = MaxProjection(
        core_axes=["height", "width"],
        iter_axis="frame",
        local_max_radius=8,
        intensity_threshold=1,
    )

    seeds = detector.fit_transform(video)

    # Create visualization
    fig = visualize_detection(
        video=video,
        seeds=seeds,
        ground_truth=positions,
        title="MaxProjection Detection Results",
    )

    # Basic figure checks
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1

    # Optional: save for manual inspection
    artifact_directory = Path("../../../artifacts")
    artifact_directory.mkdir(exist_ok=True)
    fig.savefig(artifact_directory / "detection_results.png")
