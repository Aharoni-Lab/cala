import pytest

from cala.streaming.components import SpatialInitializer, SpatialInitializerParams
from cala.streaming.core import Estimates
from tests.fixtures import stabilized_video


class TestStreamingSpatialInitializer:
    @pytest.fixture
    def default_parameters(self):
        return SpatialInitializerParams()

    @pytest.fixture
    def default_initializer(self, default_parameters):
        return SpatialInitializer(params=default_parameters)

    def test_first_frame(self, default_initializer, stabilized_video):
        video, _, _ = stabilized_video
        frame_dimensions = tuple(video.sizes[d] for d in ["width", "height"])
        estimates = Estimates(frame_dimensions)

        estimates = default_initializer.learn_one(
            estimates=estimates,
            frame=video.values[0],
        ).transform_one(estimates)
        assert estimates.spatial_footprints[0].shape == video[0].shape
