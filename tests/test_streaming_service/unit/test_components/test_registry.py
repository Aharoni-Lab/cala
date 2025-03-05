import pytest

from cala.streaming.core.components.categories import FluorescentObject
from cala.streaming.core.components.registry import Registry


class MockFluorescentObject(FluorescentObject):
    """Mock class for testing ComponentRegistry."""

    pass


class TestComponentRegistry:
    @pytest.fixture
    def empty_registry(self):
        """Create an empty ComponentRegistry for testing."""
        return Registry()

    @pytest.fixture
    def mock_component(self):
        """Create a mock component for testing."""
        return MockFluorescentObject(detected_frame_idx=0)

    @pytest.fixture
    def populated_registry(self, empty_registry, mock_component):
        """Create a registry populated with test components."""
        components = [MockFluorescentObject(detected_frame_idx=i) for i in range(3)]
        for component in components:
            empty_registry.add(component)
        return empty_registry, components

    def test_initialization(self, empty_registry):
        """Test basic initialization of ComponentRegistry."""
        assert empty_registry.n_components == 0
        assert empty_registry.ids == set()
        assert empty_registry._components == {}

    def test_add_component(self, empty_registry, mock_component):
        """Test adding a component."""
        empty_registry.add(mock_component)

        assert empty_registry.n_components == 1
        assert mock_component.id in empty_registry.ids
        assert empty_registry._components[mock_component.id] == mock_component

    def test_remove_component(self, populated_registry):
        """Test removing a component."""
        registry, components = populated_registry
        component_to_remove = components[0]

        # Test successful removal
        removed = registry.remove(component_to_remove.id)
        assert removed == component_to_remove
        assert component_to_remove.id not in registry.ids
        assert registry.n_components == 2

        # Test removing non-existent component
        removed = registry.remove(999)
        assert removed is None
        assert registry.n_components == 2

    def test_get_component(self, populated_registry):
        """Test getting a component by ID."""
        registry, components = populated_registry
        component = components[0]

        # Test getting existing component
        retrieved = registry.get(component.id)
        assert retrieved == component

        # Test getting non-existent component
        retrieved = registry.get(999)
        assert retrieved is None

    def test_get_by_type(self, populated_registry):
        """Test getting components by type."""
        registry, components = populated_registry

        # Test getting MockFluorescentObject components
        mock_ids = registry.get_by_type(MockFluorescentObject)
        assert set(mock_ids) == {c.id for c in components}

        # Test getting components of a different type
        class OtherType(FluorescentObject):
            pass

        other_ids = registry.get_by_type(OtherType)
        assert other_ids == []

    def test_clear(self, populated_registry):
        """Test clearing all components."""
        registry, _ = populated_registry

        registry.clear()
        assert registry.n_components == 0
        assert registry.ids == set()
        assert registry._components == {}

    def test_component_ids_property(self, populated_registry):
        """Test component_ids property returns correct set of IDs."""
        registry, components = populated_registry
        expected_ids = {component.id for component in components}

        assert registry.ids == expected_ids

    def test_n_components_property(self, populated_registry):
        """Test n_components property returns correct count."""
        registry, components = populated_registry

        assert registry.n_components == len(components)
