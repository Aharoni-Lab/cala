import numpy as np
import pytest
import zarr
from scipy import sparse

from cala.streaming.core.components.base import (
    ComponentUpdate,
    FluorescentObject,
    UpdateType,
)


def test_update_type_enum():
    """Test that UpdateType enum has expected values."""
    assert UpdateType.MODIFIED.name == "MODIFIED"
    assert UpdateType.ADDED.name == "ADDED"
    assert UpdateType.REMOVED.name == "REMOVED"


class TestComponentUpdate:
    def test_component_update_creation(self):
        """Test creating ComponentUpdate with different parameters."""
        # Basic creation with just update_type
        update = ComponentUpdate(update_type=UpdateType.ADDED)
        assert update.update_type == UpdateType.ADDED
        assert update.old_footprint is None
        assert update.old_time_trace is None

        # Creation with all parameters
        footprint = np.zeros((5, 5))
        time_trace = np.array([1, 2, 3])
        update = ComponentUpdate(
            update_type=UpdateType.MODIFIED,
            old_footprint=footprint,
            old_time_trace=time_trace,
        )
        assert update.update_type == UpdateType.MODIFIED
        assert isinstance(update.old_footprint, (np.ndarray, sparse.csr_matrix))
        assert isinstance(update.old_time_trace, (np.ndarray, zarr.Array))
        if update.old_footprint is not None:
            old_footprint_array = (
                update.old_footprint.toarray()
                if isinstance(update.old_footprint, sparse.csr_matrix)
                else update.old_footprint
            )
            assert np.array_equal(old_footprint_array, footprint)
        if update.old_time_trace is not None:
            assert np.array_equal(update.old_time_trace, time_trace)


class MockFluorescentObject(FluorescentObject):
    """Mock class for testing abstract FluorescentObject."""

    pass


class TestFluorescentObject:
    @pytest.fixture
    def basic_object(self):
        """Create a basic FluorescentObject for testing."""
        footprint = np.zeros((5, 5))
        footprint[2, 2] = 1
        time_trace = np.array([1.0, 2.0, 3.0])
        return MockFluorescentObject(footprint, time_trace)

    def test_initialization(self, basic_object):
        """Test basic initialization of FluorescentObject."""
        assert isinstance(basic_object.footprint, sparse.csr_matrix)
        assert isinstance(basic_object.time_trace, zarr.Array)
        assert basic_object.confidence_level is None
        assert basic_object.overlapping_objects == set()
        assert basic_object.last_update.update_type == UpdateType.ADDED

    def test_footprint_property(self, basic_object):
        """Test footprint getter and setter."""
        new_footprint = np.zeros((5, 5))
        new_footprint[0, 0] = 1

        # Store old footprint for comparison
        old_footprint = basic_object.footprint.copy()

        # Update footprint
        basic_object.footprint = new_footprint

        # Check new footprint
        assert isinstance(basic_object.footprint, sparse.csr_matrix)
        assert np.array_equal(basic_object.footprint.toarray(), new_footprint)

        # Check update was tracked
        assert basic_object.last_update.update_type == UpdateType.MODIFIED
        assert basic_object.last_update.old_footprint is not None
        assert np.array_equal(
            basic_object.last_update.old_footprint.toarray(), old_footprint.toarray()
        )

    def test_time_trace_property(self, basic_object):
        """Test time_trace getter and setter."""
        new_time_trace = np.array([4.0, 5.0, 6.0])

        # Store old time trace for comparison
        old_time_trace = basic_object.time_trace[:]

        # Update time trace
        basic_object.time_trace = new_time_trace

        # Check new time trace
        assert isinstance(basic_object.time_trace, zarr.Array)
        assert np.array_equal(basic_object.time_trace[:], new_time_trace)

        # Check update was tracked
        assert basic_object.last_update.update_type == UpdateType.MODIFIED
        assert basic_object.last_update.old_time_trace is not None
        assert np.array_equal(
            basic_object.last_update.old_time_trace[:], old_time_trace
        )

    def test_update_methods(self, basic_object):
        """Test update methods for footprint, time_trace, confidence, and overlapping objects."""
        # Test update_footprint
        new_footprint = np.ones((5, 5))
        basic_object.update_footprint(new_footprint)
        assert np.array_equal(basic_object.footprint.toarray(), new_footprint)
        assert basic_object.last_update.update_type == UpdateType.MODIFIED

        # Test update_time_trace
        new_time_trace = np.array([7.0, 8.0, 9.0])
        basic_object.update_time_trace(new_time_trace)
        assert np.array_equal(basic_object.time_trace[:], new_time_trace)
        assert basic_object.last_update.update_type == UpdateType.MODIFIED

        # Test update_confidence_level
        basic_object.update_confidence_level(0.95)
        assert basic_object.confidence_level == 0.95

        # Test update_overlapping_objects
        other_object = MockFluorescentObject(np.zeros((5, 5)), np.array([1.0, 2.0]))
        basic_object.update_overlapping_objects({other_object})
        assert basic_object.overlapping_objects == {other_object}

    def test_sparse_matrix_input(self):
        """Test initialization with sparse matrix input."""
        sparse_footprint = sparse.csr_matrix(([1], ([2], [2])), shape=(5, 5))
        obj = MockFluorescentObject(sparse_footprint, np.array([1.0, 2.0]))
        assert isinstance(obj.footprint, sparse.csr_matrix)
        assert np.array_equal(obj.footprint.toarray(), sparse_footprint.toarray())

    def test_zarr_array_input(self):
        """Test initialization with zarr array input."""
        zarr_time_trace = zarr.array([1.0, 2.0, 3.0])
        obj = MockFluorescentObject(np.zeros((5, 5)), zarr_time_trace)
        assert isinstance(obj.time_trace, zarr.Array)
        assert np.array_equal(obj.time_trace[:], zarr_time_trace[:])
