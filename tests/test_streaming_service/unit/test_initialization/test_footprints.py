import pytest

from cala.streaming.core.components import ComponentBigDaddy
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
        components = ComponentBigDaddy()

        for frame in video[0:1]:
            components = footprints_initializer.learn_transform_one(
                components=components,
                X=frame,
            )

        assert components.footprints[0].shape == video[0].shape
