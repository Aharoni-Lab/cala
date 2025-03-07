from uuid import UUID

import pytest

from cala.streaming.core.components.registry import Registry
from cala.streaming.types.types import FluorescentObject, Neuron, Background


class MockComponent(FluorescentObject):
    """Mock component class for testing."""

    pass


class TestRegistry:
    """Test suite for the Registry class."""

    @pytest.fixture
    def empty_registry(self):
        """Create an empty Registry for testing."""
        return Registry()

    @pytest.fixture
    def populated_registry(self):
        """Create a Registry populated with test components."""
        registry = Registry()
        neuron_ids = registry.create_many(2, Neuron)
        background_ids = registry.create_many(3, Background)
        return registry, neuron_ids, background_ids

    def test_registry_initialization(self, empty_registry):
        """Test basic initialization of Registry."""
        assert empty_registry.n_components == 0
        assert empty_registry.ids == []
        assert empty_registry.type_to_ids == {}
        assert empty_registry.id_to_type == {}

    def test_create_component(self, empty_registry):
        """Test creating a single component."""
        component_id = empty_registry.create(MockComponent)

        # Verify ID format and registration
        assert isinstance(UUID(component_id, version=4), UUID)  # Valid UUID4
        assert empty_registry.n_components == 1
        assert component_id in empty_registry.ids
        assert empty_registry.id_to_type[component_id] == MockComponent
        assert component_id in empty_registry.type_to_ids[MockComponent]

    def test_create_many_components(self, empty_registry):
        """Test creating multiple components at once."""
        count = 5
        component_ids = empty_registry.create_many(count, MockComponent)

        assert len(component_ids) == count
        assert empty_registry.n_components == count

        # Verify all IDs are valid and properly registered
        for component_id in component_ids:
            assert isinstance(UUID(component_id, version=4), UUID)
            assert component_id in empty_registry.ids
            assert empty_registry.id_to_type[component_id] == MockComponent
            assert component_id in empty_registry.type_to_ids[MockComponent]

    def test_remove_component(self, populated_registry):
        """Test removing a component."""
        registry, neuron_ids, background_ids = populated_registry
        id_to_remove = neuron_ids[0]

        # Remove component and verify
        registry.remove(id_to_remove)
        assert id_to_remove not in registry.ids
        assert id_to_remove not in registry.id_to_type
        assert id_to_remove not in registry.type_to_ids[Neuron]
        assert registry.n_components == len(neuron_ids) + len(background_ids) - 1

        # Test removing non-existent component
        with pytest.raises(KeyError):
            registry.remove("nonexistent_id")

    def test_get_type_by_id(self, populated_registry):
        """Test getting component type by ID."""
        registry, neuron_ids, background_ids = populated_registry

        # Test getting existing components
        assert registry.get_type_by_id(neuron_ids[0]) == Neuron
        assert registry.get_type_by_id(background_ids[0]) == Background

        # Test getting non-existent component
        with pytest.raises(KeyError):
            registry.get_type_by_id("nonexistent_id")

    def test_get_id_by_type(self, populated_registry):
        """Test getting component IDs by type."""
        registry, neuron_ids, background_ids = populated_registry

        # Test getting existing component types
        assert set(registry.get_id_by_type(Neuron)) == set(neuron_ids)
        assert set(registry.get_id_by_type(Background)) == set(background_ids)

        # Test getting non-existent component type
        assert registry.get_id_by_type(MockComponent) == []

    def test_clear_registry(self, populated_registry):
        """Test clearing all components from registry."""
        registry, _, _ = populated_registry

        registry.clear()
        assert registry.n_components == 0
        assert registry.ids == []
        assert registry.type_to_ids == {}
        assert registry.id_to_type == {}

    def test_registry_properties(self, populated_registry):
        """Test registry properties."""
        registry, neuron_ids, background_ids = populated_registry

        # Test n_components
        assert registry.n_components == len(neuron_ids) + len(background_ids)

        # Test ids
        assert set(registry.ids) == set(neuron_ids + background_ids)

    def test_component_type_isolation(self, empty_registry):
        """Test that components of different types are properly isolated."""
        # Create components of different types
        neuron_id = empty_registry.create(Neuron)
        background_id = empty_registry.create(Background)

        # Verify type isolation
        assert neuron_id in empty_registry.get_id_by_type(Neuron)
        assert neuron_id not in empty_registry.get_id_by_type(Background)
        assert background_id in empty_registry.get_id_by_type(Background)
        assert background_id not in empty_registry.get_id_by_type(Neuron)

    class TestEdgeCases:
        """Nested test class for edge cases and error conditions."""

        def test_create_with_invalid_type(self, empty_registry):
            """Test creating a component with an invalid type."""

            class InvalidComponent:  # Not a FluorescentObject
                pass

            with pytest.raises(TypeError):
                empty_registry.create(InvalidComponent)

        def test_create_many_with_zero_count(self, empty_registry):
            """Test creating zero components."""
            ids = empty_registry.create_many(0, MockComponent)
            assert ids == []
            assert empty_registry.n_components == 0

        def test_create_many_with_negative_count(self, empty_registry):
            """Test creating negative number of components."""
            with pytest.raises(ValueError):
                empty_registry.create_many(-1, MockComponent)

        def test_remove_last_component_of_type(self, empty_registry):
            """Test removing the last component of a type."""
            component_id = empty_registry.create(MockComponent)
            empty_registry.remove(component_id)

            assert empty_registry.type_to_ids[MockComponent] == []
            assert empty_registry.get_id_by_type(MockComponent) == []
