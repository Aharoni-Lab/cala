import pytest
from pathlib import Path

from tests.fixtures import raw_calcium_video, preprocessed_video, stabilized_video
from tests.visual_helper import VisualHelper


def test_e2e(raw_calcium_video, preprocessed_video, stabilized_video):
    artifact_directory = Path(__file__).parent / "artifacts"
    artifact_directory.mkdir(exist_ok=True)

    visual_helper = VisualHelper()
    visual_helper.write_movie(
        raw_calcium_video, artifact_directory / "raw_calcium_video.mp4"
    )
    visual_helper.write_movie(
        preprocessed_video, artifact_directory / "preprocessed_video.mp4"
    )
    visual_helper.write_movie(
        stabilized_video, artifact_directory / "stabilized_video.mp4"
    )
    visual_helper.test_visualize_calcium_traces(
        raw_calcium_video, artifact_directory / "calcium_traces.png"
    )
