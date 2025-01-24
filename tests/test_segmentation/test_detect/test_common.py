from matplotlib.figure import Figure
from pathlib import Path
import pytest

from .conftest import visualize_detection


def test_visualization(stabilized_video, detector_instance):
    """Test visualization of detection results."""
    video, ground_truth, _ = stabilized_video

    seeds = detector_instance.fit_transform(video)

    # Create visualization
    fig = visualize_detection(
        video=video,
        detector=detector_instance,
        seeds=seeds,
        ground_truth=ground_truth,
        title=f"{type(detector_instance).__name__} Detection Results",
    )

    # Basic figure checks
    assert isinstance(fig, Figure)
    assert len(fig.axes) == 1

    # Optional: save for manual inspection
    artifact_directory = Path(__file__).parents[2] / "artifacts"
    artifact_directory.mkdir(exist_ok=True)
    fig.savefig(
        artifact_directory / f"{type(detector_instance).__name__}_detection_results.png"
    )
