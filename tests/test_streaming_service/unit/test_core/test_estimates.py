from cala.streaming.core import Estimates
from tests.fixtures import stabilized_video


class TestStreamingEstimates:
    def test_init_estimate(self, stabilized_video):
        video, _, _ = stabilized_video
        frame_dimensions = tuple(video.sizes[d] for d in ["width", "height"])
        return Estimates(dimensions=frame_dimensions)
