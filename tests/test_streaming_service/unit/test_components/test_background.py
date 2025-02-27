import numpy as np
import pytest
import zarr
from scipy import sparse

from cala.streaming.core.components.background import Background
from cala.streaming.core.components.neuron import Neuron


class TestBackground:
    @pytest.fixture
    def basic_background(self):
        """Create a basic Background component for testing."""
        footprint = np.zeros((5, 5))
        footprint[2:4, 2:4] = 1  # Create a 2x2 square of ones
        time_trace = np.array([1.0, 2.0, 3.0])
        return Background(footprint, time_trace)

    def test_initialization(self, basic_background):
        """Test basic initialization of Background."""
        # Test default values
        assert isinstance(basic_background.footprint, sparse.csr_matrix)
        assert isinstance(basic_background.time_trace, zarr.Array)
        assert basic_background.confidence_level == 0.0  # Default value
        assert basic_background.overlapping_objects == set()
        assert basic_background._background_type == "neuropil"  # Default value

        # Test custom initialization
        footprint = np.eye(5)
        time_trace = np.array([4.0, 5.0, 6.0])
        bg = Background(
            footprint, time_trace, confidence_level=0.8, background_type="blood_vessel"
        )
        assert bg.confidence_level == 0.8
        assert bg._background_type == "blood_vessel"
        assert np.array_equal(bg.footprint.toarray(), footprint)
        assert np.array_equal(bg.time_trace[:], time_trace)

    def test_initialization_with_overlapping_objects(self):
        """Test initialization with overlapping objects."""
        footprint1 = np.zeros((5, 5))
        footprint1[0:2, 0:2] = 1
        time_trace1 = np.array([1.0, 2.0, 3.0])
        bg1 = Background(footprint1, time_trace1)

        footprint2 = np.zeros((5, 5))
        footprint2[1:3, 1:3] = 1
        time_trace2 = np.array([4.0, 5.0, 6.0])

        # Create a background component with overlapping objects
        bg2 = Background(
            footprint2,
            time_trace2,
            overlapping_objects={bg1},
            background_type="neuropil",
        )

        assert bg1 in bg2.overlapping_objects
        assert len(bg2.overlapping_objects) == 1

    def test_different_background_types(self):
        """Test creating backgrounds with different types."""
        footprint = np.zeros((5, 5))
        time_trace = np.array([1.0, 2.0, 3.0])

        bg_types = ["neuropil", "blood_vessel", "artifact"]
        for bg_type in bg_types:
            bg = Background(footprint, time_trace, background_type=bg_type)
            assert bg._background_type == bg_type

    def test_estimate_contamination_not_implemented(self, basic_background):
        """Test that estimate_contamination raises NotImplementedError."""
        neuron = Neuron(np.zeros((5, 5)), np.array([1.0, 2.0, 3.0]))
        with pytest.raises(NotImplementedError):
            basic_background.estimate_contamination(neuron)

    def test_sparse_matrix_input(self):
        """Test initialization with sparse matrix input."""
        sparse_footprint = sparse.csr_matrix(([1], ([2], [2])), shape=(5, 5))
        time_trace = np.array([1.0, 2.0])
        bg = Background(sparse_footprint, time_trace)

        assert isinstance(bg.footprint, sparse.csr_matrix)
        assert np.array_equal(bg.footprint.toarray(), sparse_footprint.toarray())

    def test_zarr_array_input(self):
        """Test initialization with zarr array input."""
        footprint = np.zeros((5, 5))
        zarr_time_trace = zarr.array([1.0, 2.0, 3.0])
        bg = Background(footprint, zarr_time_trace)

        assert isinstance(bg.time_trace, zarr.Array)
        assert np.array_equal(bg.time_trace[:], zarr_time_trace[:])

    def test_update_methods(self, basic_background):
        """Test update methods inherited from FluorescentObject."""
        # Test update_footprint
        new_footprint = np.ones((5, 5))
        basic_background.update_footprint(new_footprint)
        assert np.array_equal(basic_background.footprint.toarray(), new_footprint)

        # Test update_time_trace
        new_time_trace = np.array([7.0, 8.0, 9.0])
        basic_background.update_time_trace(new_time_trace)
        assert np.array_equal(basic_background.time_trace[:], new_time_trace)

        # Test update_confidence_level
        basic_background.update_confidence_level(0.95)
        assert basic_background.confidence_level == 0.95
