import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import (
    FootprintStore,
    ObservableStore,
    TraceStore,
    Component,
    ComponentTypes,
)


class TestObservable:
    """Test suite for the Observable base class."""

    def test_inheritance(self):
        """Test that Observable properly inherits from DataArray."""
        data = np.random.rand(5, 5)
        observable = ObservableStore(data)
        assert isinstance(observable, xr.DataArray)
        assert isinstance(observable, ObservableStore)


class TestFootprints:
    """Test suite for the Footprints class."""

    @pytest.fixture
    def sample_footprints(self):
        """Create sample footprint data."""
        data = np.random.rand(3, 10, 10)  # 3 components, 10x10 spatial dimensions
        coords = {
            "id_": ("components", ["id0", "id1", "id2"]),
            "type_": ("components", ["neuron", "neuron", "background"]),
        }
        return FootprintStore(
            data, dims=("components", "height", "width"), coords=coords
        )

    def test_initialization(self, sample_footprints):
        """Test proper initialization of Footprints."""
        assert isinstance(sample_footprints, ObservableStore)
        assert isinstance(sample_footprints, FootprintStore)
        assert sample_footprints.dims == ("components", "height", "width")
        assert "id_" in sample_footprints.coords
        assert "type_" in sample_footprints.coords


class TestTraces:
    """Test suite for the Traces class."""

    @pytest.fixture
    def sample_traces(self):
        """Create sample temporal traces data."""
        data = np.random.rand(3, 100)  # 3 components, 100 timepoints
        coords = {
            "id_": ("components", ["id0", "id1", "id2"]),
            "type_": ("components", ["neuron", "neuron", "background"]),
        }
        return TraceStore(data, dims=("components", "frames"), coords=coords)

    def test_initialization(self, sample_traces):
        """Test proper initialization of Traces."""
        assert isinstance(sample_traces, ObservableStore)
        assert isinstance(sample_traces, TraceStore)
        assert sample_traces.dims == ("components", "frames")
        assert "id_" in sample_traces.coords
        assert "type_" in sample_traces.coords


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
