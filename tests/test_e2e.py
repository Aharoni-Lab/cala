import pytest

from tests.fixtures import raw_calcium_video, preprocessed_video, stabilized_video
from tests.visual_helper import VisualHelper


def test_e2e(raw_calcium_video, preprocessed_video, stabilized_video):
    visual_helper = VisualHelper()
    visual_helper.write_movie(raw_calcium_video, "./raw_calcium_video.mp4")
    visual_helper.write_movie(preprocessed_video, "./preprocessed_video.mp4")
    visual_helper.write_movie(stabilized_video, "./stabilized_video.mp4")
    visual_helper.test_visualize_calcium_traces(
        raw_calcium_video, "./calcium_traces.png"
    )
