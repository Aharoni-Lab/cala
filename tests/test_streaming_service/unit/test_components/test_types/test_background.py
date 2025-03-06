import pytest
from cala.streaming.core.components.categories import Background, Neuron
from cala.streaming.types.base import UpdateType

from .test_base import BaseFluorescentObjectTest


class TestBackground(BaseFluorescentObjectTest):
    @pytest.fixture
    def basic_object(self):
        """Create a basic Background component for testing."""
        return Background(detected_frame_idx=0)

    @pytest.fixture
    def custom_background(self):
        """Create a Background with custom parameters for testing."""
        return Background(
            detected_frame_idx=1, confidence_level=0.8, background_type="blood_vessel"
        )

    def test_background_specific_initialization(self, basic_object):
        """Test Background-specific initialization attributes."""
        assert basic_object.background_type == "neuropil"  # Default value

    def test_custom_initialization(self, custom_background):
        """Test initialization with custom parameters."""
        assert custom_background.detected_frame_idx == 1
        assert custom_background.confidence_level == 0.8
        assert custom_background.background_type == "blood_vessel"
        assert custom_background.last_update.update_type == UpdateType.ADDED
        assert custom_background.last_update.last_update_frame_idx == 1

    def test_different_background_types(self):
        """Test creating backgrounds with different categories."""
        bg_types = ["neuropil", "blood_vessel", "artifact"]
        for bg_type in bg_types:
            bg = Background(detected_frame_idx=0, background_type=bg_type)
            assert bg.background_type == bg_type

    def test_estimate_contamination_not_implemented(self, basic_object):
        """Test that estimate_contamination raises NotImplementedError."""
        neuron = Neuron(detected_frame_idx=0)
        with pytest.raises(NotImplementedError):
            basic_object.estimate_contamination(neuron)

    def test_update_confidence_level(self, basic_object):
        """Test confidence level updates."""
        basic_object.update_confidence_level(0.95, 2)
        assert basic_object.confidence_level == 0.95
        assert basic_object.last_update.update_type == UpdateType.MODIFIED
        assert basic_object.last_update.last_update_frame_idx == 2
