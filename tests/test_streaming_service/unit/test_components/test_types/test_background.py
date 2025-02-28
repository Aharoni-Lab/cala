import pytest

from cala.streaming.core.components.types import Background, Neuron
from cala.streaming.core.components.types.base import UpdateType


class TestBackground:
    @pytest.fixture
    def basic_background(self):
        """Create a basic Background component for testing."""
        return Background(detected_frame_idx=0)

    @pytest.fixture
    def custom_background(self):
        """Create a Background with custom parameters for testing."""
        return Background(
            detected_frame_idx=1, confidence_level=0.8, background_type="blood_vessel"
        )

    def test_initialization(self, basic_background):
        """Test basic initialization of Background."""
        assert basic_background.detected_frame_idx == 0
        assert basic_background.confidence_level is None
        assert basic_background.background_type == "neuropil"  # Default value
        assert basic_background.last_update.update_type == UpdateType.ADDED
        assert basic_background.last_update.last_update_frame_idx == 0

    def test_custom_initialization(self, custom_background):
        """Test initialization with custom parameters."""
        assert custom_background.detected_frame_idx == 1
        assert custom_background.confidence_level == 0.8
        assert custom_background.background_type == "blood_vessel"
        assert custom_background.last_update.update_type == UpdateType.ADDED
        assert custom_background.last_update.last_update_frame_idx == 1

    def test_different_background_types(self):
        """Test creating backgrounds with different types."""
        bg_types = ["neuropil", "blood_vessel", "artifact"]
        for bg_type in bg_types:
            bg = Background(detected_frame_idx=0, background_type=bg_type)
            assert bg.background_type == bg_type

    def test_estimate_contamination_not_implemented(self, basic_background):
        """Test that estimate_contamination raises NotImplementedError."""
        neuron = Neuron(detected_frame_idx=0)
        with pytest.raises(NotImplementedError):
            basic_background.estimate_contamination(neuron)

    def test_update_confidence_level(self, basic_background):
        """Test confidence level updates."""
        basic_background.update_confidence_level(0.95, 2)
        assert basic_background.confidence_level == 0.95
        assert basic_background.last_update.update_type == UpdateType.MODIFIED
        assert basic_background.last_update.last_update_frame_idx == 2
