import pytest

from cala.streaming.components import (
    SpatialInitializer,
    SpatialInitializerParams,
    TemporalInitializer,
    TemporalInitializerParams,
)
from cala.streaming.core import Estimates
from tests.fixtures import stabilized_video


class TestStreamingTemporalInitializer:
    @pytest.fixture
    def spatial_parameters(self):
        return SpatialInitializerParams()

    @pytest.fixture
    def spatial_initializer(self, spatial_parameters):
        return SpatialInitializer(params=spatial_parameters)

    @pytest.fixture
    def temporal_parameters(self):
        return TemporalInitializerParams()

    @pytest.fixture
    def temporal_initializer(self, temporal_parameters):
        return TemporalInitializer(params=temporal_parameters)

    @pytest.fixture
    def spatial_estimates(self, spatial_initializer, stabilized_video):
        video, _, _ = stabilized_video
        frame_dimensions = tuple(video.sizes[d] for d in ["width", "height"])
        default_estimates = Estimates(frame_dimensions)

        estimates = spatial_initializer.learn_one(
            estimates=default_estimates,
            frame=video.values[0],
        ).transform_one(default_estimates)

        return estimates

    def test_first_n_frames(
        self, stabilized_video, temporal_initializer, spatial_estimates
    ):
        video, _, _ = stabilized_video
        temporal_estimates = temporal_initializer.learn_one(
            spatial_estimates, video[:3]
        ).transform_one(spatial_estimates)

        assert (
            spatial_estimates.spatial_footprints.shape[0]
            == temporal_estimates.temporal_traces.shape[0]
        )
