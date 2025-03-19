import numpy as np
import pytest
import sparse
import xarray as xr

from cala.streaming.core import FootprintStore
from cala.streaming.init.odl.overlaps import (
    OverlapsInitializer,
    OverlapsInitializerParams,
)
from cala.streaming.stores.odl import OverlapStore


class TestOverlapsInitializerParams:
    """Test suite for OverlapGroupsInitializerParams."""

    @pytest.fixture
    def default_params(self):
        """Create default parameters instance."""
        return OverlapsInitializerParams()

    def test_default_values(self, default_params):
        """Test default parameter values."""
        assert default_params.component_axis == "components"
        assert default_params.id_coordinates == "id_"
        assert default_params.type_coordinates == "type_"
        assert default_params.spatial_axes == ("height", "width")

    def test_validation_valid_spatial_axes(self):
        """Test validation with valid spatial axes."""
        params = OverlapsInitializerParams(spatial_axes=("y", "x"))
        params.validate()  # Should not raise

    def test_validation_invalid_spatial_axes(self):
        """Test validation with invalid spatial axes."""
        # Test with wrong type
        with pytest.raises(ValueError):
            params = OverlapsInitializerParams(spatial_axes=["height", "width"])
            params.validate()

        # Test with wrong length
        with pytest.raises(ValueError):
            params = OverlapsInitializerParams(spatial_axes=("height",))
            params.validate()


class TestOverlapsInitializer:
    """Test suite for OverlapsInitializer."""

    @pytest.fixture
    def sample_footprints(self):
        """Create sample footprints for testing.

        Creates a set of footprints with known overlap patterns:
        - Components 0 and 1 overlap
        - Component 2 is isolated
        - Components 3 and 4 overlap
        """
        n_components = 5
        height, width = 10, 10

        # Create empty footprints
        footprints_data = np.zeros((n_components, height, width))

        # Set up specific overlap patterns
        footprints_data[0, 0:5, 0:5] = 1  # Component 0
        footprints_data[1, 3:8, 3:8] = 1  # Component 1 (overlaps with 0)
        footprints_data[2, 8:10, 8:10] = 1  # Component 2 (isolated)
        footprints_data[3, 0:3, 8:10] = 1  # Component 3
        footprints_data[4, 1:4, 7:9] = 1  # Component 4 (overlaps with 3)

        coords = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": ("components", ["neuron"] * 3 + ["background"] * 2),
        }

        return FootprintStore(
            footprints_data, dims=("components", "height", "width"), coords=coords
        )

    @pytest.fixture
    def initializer(self):
        """Create OverlapsInitializer instance."""
        return OverlapsInitializer(OverlapsInitializerParams())

    def test_initialization(self, initializer):
        """Test proper initialization."""
        assert isinstance(initializer.params, OverlapsInitializerParams)
        assert not hasattr(initializer, "overlaps_")

    def test_learn_one(self, initializer, sample_footprints):
        """Test learn_one method."""
        initializer.learn_one(sample_footprints)

        # Check that overlaps_ was created
        assert hasattr(initializer, "overlaps_")
        assert isinstance(initializer.overlaps_, xr.DataArray)

        # Check dimensions
        assert initializer.overlaps_.dims == ("components", "components")
        assert initializer.overlaps_.shape == (5, 5)

        # Check coordinates
        assert "id_" in initializer.overlaps_.coords
        assert "type_" in initializer.overlaps_.coords

    def test_transform_one(self, initializer, sample_footprints):
        """Test transform_one method."""
        initializer.learn_one(sample_footprints)
        result = initializer.transform_one()

        # Check result type
        assert isinstance(result, OverlapStore)
        assert isinstance(result.data, sparse.COO)

    def test_overlap_detection_correctness(self, initializer, sample_footprints):
        """Test the correctness of overlap detection."""
        initializer.learn_one(sample_footprints)
        result = initializer.transform_one()

        # Convert to dense for testing
        overlap_matrix = result.data.todense()

        # Test expected overlap patterns
        assert overlap_matrix[0, 1] == 1  # Components 0 and 1 overlap
        assert overlap_matrix[1, 0] == 1  # Symmetric
        assert np.sum(overlap_matrix[2]) == 1  # Component 2 only overlaps with itself
        assert overlap_matrix[1, 4] == 1  # Components 3 and 4 overlap
        assert overlap_matrix[4, 1] == 1  # Components 3 and 4 overlap
        assert overlap_matrix[3, 4] == 1  # Components 3 and 4 overlap
        assert overlap_matrix[4, 3] == 1  # Symmetric

    def test_coordinate_preservation(self, initializer, sample_footprints):
        """Test that coordinates are properly preserved."""
        initializer.learn_one(sample_footprints)
        result = initializer.transform_one()

        assert np.array_equal(
            result.coords["id_"].values, sample_footprints.coords["id_"].values
        )
        assert np.array_equal(
            result.coords["type_"].values, sample_footprints.coords["type_"].values
        )

    def test_symmetry(self, initializer, sample_footprints):
        """Test that the overlap matrix is symmetric."""
        initializer.learn_one(sample_footprints)
        result = initializer.transform_one()

        dense_matrix = result.data.todense()
        assert np.allclose(dense_matrix, dense_matrix.T)

    def test_diagonal(self, initializer, sample_footprints):
        """Test that diagonal elements are properly set."""
        initializer.learn_one(sample_footprints)
        result = initializer.transform_one()

        dense_matrix = result.data.todense()
        assert np.allclose(np.diag(dense_matrix), 1)
