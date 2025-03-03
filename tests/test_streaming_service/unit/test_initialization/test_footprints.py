import pytest

from cala.streaming.core.components import ComponentManager
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
        components = ComponentManager()

        components = footprints_initializer.learn_one(
            components=components,
            X=video[0],
        ).transform_one(components)

        assert components.footprints[0].shape == video[0].shape
