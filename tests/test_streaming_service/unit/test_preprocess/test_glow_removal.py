import pytest

from cala.streaming.nodes.preprocess import GlowRemover


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
