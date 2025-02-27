import pytest

from cala.streaming.core.components import ComponentManager
from cala.streaming.initialization import SpatialInitializer, SpatialInitializerParams
from tests.fixtures import stabilized_video


class TestStreamingSpatialInitializer:
    @pytest.fixture
    def spatial_parameters(self):
        return SpatialInitializerParams()

    @pytest.fixture
    def spatial_initializer(self, spatial_parameters):
        return SpatialInitializer(params=spatial_parameters)

    def test_first_frame(self, spatial_initializer, stabilized_video):
        video, _, _ = stabilized_video
        components = ComponentManager()

        components = spatial_initializer.learn_one(
            components=components,
            frame=video.values[0],
        ).transform_one(components)

        assert components.footprints[0].shape == video[0].shape
