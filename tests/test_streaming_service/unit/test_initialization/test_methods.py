import pytest

from cala.streaming.initialization import Initializer
from tests.fixtures import stabilized_video


class TestStreamingInitializationMethods:
    @pytest.fixture
    def default_initializer(self):
        return Initializer()

    def test_first_frame(self, default_initializer, stabilized_video):
        video, _, _ = stabilized_video

        spatial_footprint = default_initializer.watershed_components(video.values[0])
        assert spatial_footprint[0].shape == video[0].shape
