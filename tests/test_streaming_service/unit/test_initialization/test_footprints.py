import pytest

from cala.streaming.initialization import (
    FootprintsInitializer,
    FootprintsInitializerParams,
)
from tests.fixtures import stabilized_video


class TestStreamingFootprintsInitializer:
    @pytest.fixture
    def footprints_parameters(self):
        return FootprintsInitializerParams()

    @pytest.fixture
    def footprints_initializer(self, footprints_parameters):
        return FootprintsInitializer(params=footprints_parameters)

    def test_first_frame(self, footprints_initializer, stabilized_video):
        video, _, _ = stabilized_video

        for frame in video[0:1]:
            footprints_initializer.learn_one(
                frame=frame,
            )
            neuron_footprints, background_footprints = (
                footprints_initializer.transform_one()
            )

        assert neuron_footprints[0].shape == video[0].shape
        assert background_footprints[0].shape == video[0].shape
