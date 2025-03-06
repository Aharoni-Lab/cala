import numpy as np
import pytest
import xarray as xr

from cala.streaming.core.components.stores.footprints import FootprintStore


class TestFootprintManager:
    @pytest.fixture
    def basic_manager(self):
        """Create a basic FootprintManager for testing."""
        return FootprintStore()

    @pytest.fixture
    def sample_footprint(self):
        """Create a sample footprint for testing."""
        data = np.zeros((10, 10))
        data[4:7, 4:7] = 1  # 3x3 square of ones
        return xr.DataArray(
            data,
            dims=["height", "width"],
            coords={"height": range(10), "width": range(10)},
        )

    @pytest.fixture
    def initialized_manager(self, basic_manager, sample_footprint):
        """Create a FootprintManager initialized with a sample footprint."""
        footprints = sample_footprint.expand_dims(dim={"component": [1]})
        basic_manager.initialize(footprints)
        return basic_manager

    def test_initialization(self, basic_manager):
        """Test basic initialization of FootprintManager."""
        assert basic_manager.component_axis == "component"
        assert basic_manager.spatial_axes == ("width", "height")
        assert basic_manager.footprints_dimensions == ("component", "width", "height")

    def test_initialize_with_data(self, basic_manager, sample_footprint):
        """Test initializing with footprint data."""
        # Create a DataArray with proper dimensions
        footprints = sample_footprint.expand_dims(dim={"component": [1]})
        basic_manager.initialize(footprints)

        assert hasattr(basic_manager, "_footprints")
        assert isinstance(basic_manager.footprints, xr.DataArray)
        assert set(basic_manager.footprints.dims) == {"component", "width", "height"}

    def test_initialize_with_wrong_dimensions(self, basic_manager, sample_footprint):
        """Test initialization with incorrect dimensions raises error."""
        with pytest.raises(ValueError, match="Footprints dimensions must be"):
            basic_manager.initialize(sample_footprint)  # Missing component dimension

    def test_add_footprint(self, initialized_manager, sample_footprint):
        """Test adding a new footprint."""
        # Create a different footprint
        new_footprint = sample_footprint.copy()
        new_footprint.values[1:4, 1:4] = 1

        initialized_manager.add_footprint(2, new_footprint)

        # Check footprint was added
        assert 2 in initialized_manager.footprints.component.values
        assert len(initialized_manager.footprints.component) == 2

        # Check overlap tracking was initialized
        assert 2 in initialized_manager._overlapping_components

    def test_remove_footprint(self, initialized_manager):
        """Test removing a footprint."""
        initialized_manager.remove_footprint(1)

        # Check footprint was removed
        assert 1 not in initialized_manager.footprints.component.values
        assert len(initialized_manager.footprints.component) == 0

        # Check overlap tracking was cleaned up
        assert 1 not in initialized_manager._overlapping_components

    def test_update_footprint(self, initialized_manager, sample_footprint):
        """Test updating an existing footprint."""
        # Modify the footprint
        updated_footprint = sample_footprint.copy()
        updated_footprint.values[1:4, 1:4] = 2

        initialized_manager.update_footprint(1, updated_footprint)

        # Check footprint was updated
        np.testing.assert_array_equal(
            initialized_manager.footprints.sel(component=1).values,
            updated_footprint.values,
        )

    def test_overlapping_components(self, initialized_manager, sample_footprint):
        """Test detection of overlapping components."""
        # Create an overlapping footprint
        overlap_footprint = xr.zeros_like(sample_footprint)
        overlap_footprint.values[5:8, 5:8] = 1  # Overlaps with original footprint

        # Create a non-overlapping footprint
        non_overlap_footprint = xr.zeros_like(sample_footprint)
        non_overlap_footprint.values[0:3, 0:3] = 1  # No overlap

        # Add both footprints
        initialized_manager.add_footprint(2, overlap_footprint)
        initialized_manager.add_footprint(3, non_overlap_footprint)

        # Check overlapping relationships
        assert initialized_manager.get_overlapping_components(1) == {2}
        assert initialized_manager.get_overlapping_components(2) == {1}
        assert initialized_manager.get_overlapping_components(3) == set()

    def test_get_overlapping_components_nonexistent(self, initialized_manager):
        """Test getting overlapping components for nonexistent component."""
        assert initialized_manager.get_overlapping_components(999) is None

    def test_custom_axes(self):
        """Test using custom axis names."""
        manager = FootprintStore(component_axis="neuron", spatial_axes=("y", "x"))
        assert manager.component_axis == "neuron"
        assert manager.spatial_axes == ("y", "x")
        assert manager.footprints_dimensions == ("neuron", "y", "x")
