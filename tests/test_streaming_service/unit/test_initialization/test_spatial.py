import pytest

from cala.streaming.core import Estimates
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
        frame_dimensions = tuple(video.sizes[d] for d in ["width", "height"])
        estimates = Estimates(frame_dimensions)

        estimates = spatial_initializer.learn_one(
            estimates=estimates,
            frame=video.values[0],
        ).transform_one(estimates)
        assert estimates.spatial_footprints[0].shape == video[0].shape
