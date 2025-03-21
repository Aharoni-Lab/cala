import numpy as np
import pytest

from cala.streaming.core import (
    ObservableStore,
    Component,
    ComponentTypes,
)


class TestObservable:
    """Test suite for the Observable base class."""

    def test_inheritance(self):
        """Test that Observable properly inherits from DataArray."""
        data = np.random.rand(5, 5)
        observable = ObservableStore(data)
        assert isinstance(observable, ObservableStore)


class TestComponent:
    """Test suite for the Component enumeration."""

    def test_values(self):
        """Test Component enum values."""
        assert Component.NEURON.value == "neuron"
        assert Component.BACKGROUND.value == "background"

    def test_membership(self):
        """Test Component enum membership."""
        assert Component("neuron") == Component.NEURON
        assert Component("background") == Component.BACKGROUND

    def test_invalid_value(self):
        """Test that invalid values raise ValueError."""
        with pytest.raises(ValueError):
            Component("invalid")


class TestComponentTypes:
    """Test suite for the ComponentTypes list class."""

    def test_initialization(self):
        """Test initialization with valid components."""
        components = ComponentTypes([Component.NEURON, Component.BACKGROUND])
        assert len(components) == 2
        assert components[0] == Component.NEURON
        assert components[1] == Component.BACKGROUND

    def test_initialization_empty(self):
        """Test empty initialization."""
        components = ComponentTypes()
        assert len(components) == 0

    def test_append(self):
        """Test append functionality."""
        components = ComponentTypes()
        components.append(Component.NEURON)
        assert len(components) == 1
        assert components[0] == Component.NEURON

        with pytest.raises(ValueError):
            components.append("invalid")

    def test_extend(self):
        """Test extend functionality."""
        components = ComponentTypes()
        components.extend([Component.NEURON, Component.BACKGROUND])
        assert len(components) == 2

        with pytest.raises(ValueError):
            components.extend([Component.NEURON, "invalid"])

    def test_addition(self):
        """Test addition operations."""
        comp1 = ComponentTypes([Component.NEURON])
        comp2 = ComponentTypes([Component.BACKGROUND])

        # Test __add__
        result = comp1 + comp2
        assert len(result) == 2
        assert isinstance(result, ComponentTypes)

        # Test __iadd__
        comp1 += comp2
        assert len(comp1) == 2

    def test_setitem(self):
        """Test setting items."""
        components = ComponentTypes([Component.NEURON])

        # Test single item
        components[0] = Component.BACKGROUND
        assert components[0] == Component.BACKGROUND

        # Test slice
        components = ComponentTypes([Component.NEURON, Component.BACKGROUND])
        components[0:2] = [Component.BACKGROUND, Component.NEURON]
        assert components[0] == Component.BACKGROUND
        assert components[1] == Component.NEURON

        with pytest.raises(ValueError):
            components[0] = "invalid"

    def test_invalid_operations(self):
        """Test invalid operations raise appropriate errors."""
        components = ComponentTypes()

        with pytest.raises(ValueError):
            components.insert(0, "invalid")

        with pytest.raises(ValueError):
            components.extend(["invalid"])
