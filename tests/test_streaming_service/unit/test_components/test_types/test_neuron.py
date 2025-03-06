import pytest
from cala.streaming.core.components.categories import Neuron
from cala.streaming.types.base import UpdateType

from .test_base import BaseFluorescentObjectTest


class TestNeuron(BaseFluorescentObjectTest):
    @pytest.fixture
    def basic_object(self):
        """Create a basic Neuron for testing."""
        return Neuron(detected_frame_idx=0)

    @pytest.fixture
    def complex_neuron(self):
        """Create a Neuron with all optional parameters for testing."""
        metadata = {"location": "layer_2/3", "quality": "good"}
        return Neuron(
            detected_frame_idx=1,
            confidence_level=0.9,
            cell_type="pyramidal",
            metadata=metadata,
            rise_time_constant=0.1,
            decay_time_constant=0.5,
        )

    def test_neuron_specific_initialization(self, basic_object):
        """Test Neuron-specific initialization attributes."""
        assert basic_object.cell_type is None
        assert basic_object.metadata == {}
        assert basic_object.rise_time_constant is None
        assert basic_object.decay_time_constant is None

    def test_complex_initialization(self, complex_neuron):
        """Test initialization with all optional parameters."""
        assert complex_neuron.detected_frame_idx == 1
        assert complex_neuron.confidence_level == 0.9
        assert complex_neuron.cell_type == "pyramidal"
        assert complex_neuron.metadata["location"] == "layer_2/3"
        assert complex_neuron.metadata["quality"] == "good"
        assert complex_neuron.rise_time_constant == 0.1
        assert complex_neuron.decay_time_constant == 0.5
        assert complex_neuron.last_update.update_type == UpdateType.ADDED
        assert complex_neuron.last_update.last_update_frame_idx == 1

    def test_metadata_updates(self, basic_object):
        """Test metadata updates."""
        # Test setting initial metadata
        metadata = {"type": "pyramidal", "layer": "2/3"}
        basic_object.metadata = metadata
        assert basic_object.metadata == metadata

        # Test metadata updates
        basic_object.metadata["quality"] = "good"
        assert "quality" in basic_object.metadata
        assert basic_object.metadata["quality"] == "good"

    def test_not_implemented_methods(self, basic_object):
        """Test that unimplemented methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            basic_object.classify_cell_type()

    def test_update_confidence_level(self, basic_object):
        """Test confidence level updates."""
        basic_object.update_confidence_level(0.95, 2)
        assert basic_object.confidence_level == 0.95
        assert basic_object.last_update.update_type == UpdateType.MODIFIED
        assert basic_object.last_update.last_update_frame_idx == 2
