import numpy as np
import pytest

from cala.streaming.preprocess import GlowRemover
from tests.fixtures import raw_calcium_video


class TestStreamingGlowRemover:
    @pytest.fixture
    def glow_remover(self):
        """Create GlowRemover instance"""
        return GlowRemover()

    def test_initialization(
        self,
    ):
        """Test proper initialization of GlowRemover"""
        glow_remover = GlowRemover()
        assert isinstance(glow_remover, GlowRemover)

    def test_streaming_consistency(self, glow_remover, raw_calcium_video):
        """Test consistency of streaming glow removal"""
        video, _, _ = raw_calcium_video

        # Process frames sequentially
        streaming_results = []
        for frame in video:
            glow_remover.learn_one(frame)
            streaming_results.append(glow_remover.transform_one(frame))

        # Process frames in batch
        base_brightness = video.min("frames").compute()

        batch_results = video - base_brightness

        # Compare results
        np.testing.assert_array_almost_equal(
            streaming_results[-1], batch_results.isel({"frames": -1})
        )
