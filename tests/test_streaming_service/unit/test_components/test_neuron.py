import numpy as np
import pytest
import zarr
from scipy import sparse

from cala.streaming.core.components.types import Neuron


class TestNeuron:
    @pytest.fixture
    def basic_neuron(self):
        """Create a basic Neuron for testing."""
        footprint = np.zeros((5, 5))
        footprint[2:4, 2:4] = 1  # Create a 2x2 square of ones
        time_trace = np.array([1.0, 2.0, 3.0])
        return Neuron(footprint, time_trace)

    @pytest.fixture
    def complex_neuron(self):
        """Create a Neuron with all optional parameters for testing."""
        footprint = np.eye(5)
        time_trace = np.array([1.0, 2.0, 3.0])
        deconvolved_signal = np.array([0.0, 1.0, 0.0])
        spike_times = np.array([1])
        metadata = {"location": "layer_2/3", "quality": "good"}

        return Neuron(
            footprint=footprint,
            time_trace=time_trace,
            confidence_level=0.9,
            deconvolved_signal=deconvolved_signal,
            spike_times=spike_times,
            cell_type="pyramidal",
            metadata=metadata,
            rise_time_constant=0.1,
            decay_time_constant=0.5,
        )

    def test_initialization(self, basic_neuron):
        """Test basic initialization of Neuron."""
        # Test default values
        assert isinstance(basic_neuron.footprint, sparse.csr_matrix)
        assert isinstance(basic_neuron.time_trace, zarr.Array)
        assert basic_neuron.confidence_level == 0.0
        assert basic_neuron.overlapping_objects == set()
        assert basic_neuron.deconvolved_signal is None
        assert basic_neuron.spike_times is None
        assert basic_neuron._cell_type is None
        assert basic_neuron._metadata == {}
        assert basic_neuron._rise_time_constant is None
        assert basic_neuron._decay_time_constant is None

    def test_complex_initialization(self, complex_neuron):
        """Test initialization with all optional parameters."""
        assert complex_neuron.confidence_level == 0.9
        assert complex_neuron._cell_type == "pyramidal"
        assert "location" in complex_neuron.metadata
        assert complex_neuron.metadata["quality"] == "good"
        assert complex_neuron._rise_time_constant == 0.1
        assert complex_neuron._decay_time_constant == 0.5
        assert np.array_equal(
            complex_neuron.deconvolved_signal, np.array([0.0, 1.0, 0.0])
        )
        assert np.array_equal(complex_neuron.spike_times, np.array([1]))

    def test_initialization_with_overlapping_objects(self):
        """Test initialization with overlapping objects."""
        footprint1 = np.zeros((5, 5))
        footprint1[0:2, 0:2] = 1
        time_trace1 = np.array([1.0, 2.0, 3.0])
        neuron1 = Neuron(footprint1, time_trace1)

        footprint2 = np.zeros((5, 5))
        footprint2[1:3, 1:3] = 1
        time_trace2 = np.array([4.0, 5.0, 6.0])

        # Create a neuron with overlapping objects
        neuron2 = Neuron(footprint2, time_trace2, overlapping_objects={neuron1})

        assert neuron1 in neuron2.overlapping_objects
        assert len(neuron2.overlapping_objects) == 1

    def test_deconvolved_signal_property(self, basic_neuron):
        """Test deconvolved_signal getter and setter."""
        signal = np.array([0.0, 1.0, 0.0])
        basic_neuron.deconvolved_signal = signal
        assert np.array_equal(basic_neuron.deconvolved_signal, signal)

        # Test setting to None
        basic_neuron.deconvolved_signal = None
        assert basic_neuron.deconvolved_signal is None

    def test_spike_times_property(self, basic_neuron):
        """Test spike_times getter and setter."""
        spikes = np.array([1, 3, 5])
        basic_neuron.spike_times = spikes
        assert np.array_equal(basic_neuron.spike_times, spikes)

        # Test setting to None
        basic_neuron.spike_times = None
        assert basic_neuron.spike_times is None

    def test_metadata_property(self, basic_neuron):
        """Test metadata getter and setter."""
        metadata = {"type": "pyramidal", "layer": "2/3"}
        basic_neuron.metadata = metadata
        assert basic_neuron.metadata == metadata

        # Test metadata updates
        basic_neuron.metadata["quality"] = "good"
        assert "quality" in basic_neuron.metadata
        assert basic_neuron.metadata["quality"] == "good"

    def test_sparse_matrix_input(self):
        """Test initialization with sparse matrix input."""
        sparse_footprint = sparse.csr_matrix(([1], ([2], [2])), shape=(5, 5))
        time_trace = np.array([1.0, 2.0])
        neuron = Neuron(sparse_footprint, time_trace)

        assert isinstance(neuron.footprint, sparse.csr_matrix)
        assert np.array_equal(neuron.footprint.toarray(), sparse_footprint.toarray())

    def test_zarr_array_input(self):
        """Test initialization with zarr array input."""
        footprint = np.zeros((5, 5))
        zarr_time_trace = zarr.array([1.0, 2.0, 3.0])
        neuron = Neuron(footprint, zarr_time_trace)

        assert isinstance(neuron.time_trace, zarr.Array)
        assert np.array_equal(neuron.time_trace[:], zarr_time_trace[:])

    def test_not_implemented_methods(self, basic_neuron):
        """Test that unimplemented methods raise NotImplementedError."""
        with pytest.raises(NotImplementedError):
            basic_neuron.deconvolve_signal()

        with pytest.raises(NotImplementedError):
            basic_neuron.detect_spikes()

        with pytest.raises(NotImplementedError):
            basic_neuron.classify_cell_type()

    def test_update_methods(self, basic_neuron):
        """Test update methods inherited from FluorescentObject."""
        # Test update_footprint
        new_footprint = np.ones((5, 5))
        basic_neuron.update_footprint(new_footprint)
        assert np.array_equal(basic_neuron.footprint.toarray(), new_footprint)

        # Test update_time_trace
        new_time_trace = np.array([7.0, 8.0, 9.0])
        basic_neuron.update_time_trace(new_time_trace)
        assert np.array_equal(basic_neuron.time_trace[:], new_time_trace)

        # Test update_confidence_level
        basic_neuron.update_confidence_level(0.95)
        assert basic_neuron.confidence_level == 0.95
