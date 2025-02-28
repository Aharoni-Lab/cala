import pytest

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
        # Basic creation
        update = ComponentUpdate(update_type=UpdateType.ADDED, last_update_frame_idx=0)
        assert update.update_type == UpdateType.ADDED
        assert update.last_update_frame_idx == 0

        # Creation with all parameters
        update = ComponentUpdate(
            update_type=UpdateType.MODIFIED,
            last_update_frame_idx=1,
        )
        assert update.update_type == UpdateType.MODIFIED
        assert update.last_update_frame_idx == 1


class MockFluorescentObject(FluorescentObject):
    """Mock class for testing abstract FluorescentObject."""

    pass


class TestFluorescentObject:
    @pytest.fixture
    def basic_object(self):
        """Create a basic FluorescentObject for testing."""
        return MockFluorescentObject(detected_frame_idx=0)

    def test_initialization(self, basic_object):
        """Test basic initialization of FluorescentObject."""
        assert basic_object.confidence_level is None
        assert basic_object.overlapping_objects == set()
        assert basic_object.last_update.update_type == UpdateType.ADDED

    def test_update_methods(self, basic_object):
        """Test update methods for footprint, time_trace, confidence, and overlapping objects."""
        # Test update_overlapping_objects
        basic_object.update_confidence_level(0.95, 2)
        assert basic_object.confidence_level == 0.95
        assert basic_object.last_update.update_type == UpdateType.MODIFIED
        assert basic_object.last_update.last_update_frame_idx == 2

        # Test update_overlapping_objects
        other_object = MockFluorescentObject(detected_frame_idx=1)
        basic_object.update_overlapping_objects({other_object}, 2)
        assert basic_object.overlapping_objects == {other_object}
