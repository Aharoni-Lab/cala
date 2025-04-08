import numpy as np
import pytest
import sparse
import xarray as xr

from cala.streaming.nodes.init.odl.overlaps import (
    OverlapsInitializer,
    OverlapsInitializerParams,
)
from cala.viz_util import Visualizer


class TestOverlapsInitializer:
    """Test suite for OverlapsInitializer."""

    @pytest.fixture
    def initializer(self) -> OverlapsInitializer:
        """Create OverlapsInitializer instance."""
        return OverlapsInitializer(OverlapsInitializerParams())

    def test_initialization(self, initializer: OverlapsInitializer) -> None:
        """Test proper initialization."""
        assert isinstance(initializer.params, OverlapsInitializerParams)
        assert not hasattr(initializer, "overlaps_")

    def test_learn_one(
        self, initializer: OverlapsInitializer, mini_footprints: xr.DataArray
    ) -> None:
        """Test learn_one method."""
        initializer.learn_one(mini_footprints)

        # Check that overlaps_ was created
        assert hasattr(initializer, "overlaps_")
        assert isinstance(initializer.overlaps_, xr.DataArray)

        # Check dimensions
        assert initializer.overlaps_.dims == ("component", "component'")
        assert initializer.overlaps_.shape == (5, 5)

        # Check coordinates
        assert "id_" in initializer.overlaps_.coords
        assert "type_" in initializer.overlaps_.coords

    def test_transform_one(
        self, initializer: OverlapsInitializer, mini_footprints: xr.DataArray
    ) -> None:
        """Test transform_one method."""
        initializer.learn_one(mini_footprints)
        result = initializer.transform_one()

        # Check result type
        assert isinstance(result.data, sparse.COO)

    @pytest.mark.viz
    def test_overlap_detection_correctness(
        self,
        initializer: OverlapsInitializer,
        mini_footprints: xr.DataArray,
        visualizer: Visualizer,
    ) -> None:
        """Test the correctness of overlap detection."""
        visualizer.plot_footprints(mini_footprints, subdir="init/overlap")

        initializer.learn_one(mini_footprints)
        result = initializer.transform_one()

        result.values = result.data.todense()
        visualizer.plot_overlaps(result, mini_footprints, subdir="init/overlap")
        # Convert to dense for testing

        # Test expected overlap patterns
        assert result[0, 1] == 1  # Components 0 and 1 overlap
        assert result[1, 0] == 1  # Symmetric
        assert np.sum(result[2]) == 1  # Component 2 only overlaps with itself
        assert result[1, 4] == 1  # Components 3 and 4 overlap
        assert result[4, 1] == 1  # Components 3 and 4 overlap
        assert result[3, 4] == 1  # Components 3 and 4 overlap
        assert result[4, 3] == 1  # Symmetric

    def test_coordinate_preservation(
        self, initializer: OverlapsInitializer, mini_footprints: xr.DataArray
    ) -> None:
        """Test that coordinates are properly preserved."""
        initializer.learn_one(mini_footprints)
        result = initializer.transform_one()

        assert np.array_equal(result.coords["id_"].values, mini_footprints.coords["id_"].values)
        assert np.array_equal(result.coords["type_"].values, mini_footprints.coords["type_"].values)

    def test_symmetry(
        self, initializer: OverlapsInitializer, mini_footprints: xr.DataArray
    ) -> None:
        """Test that the overlap matrix is symmetric."""
        initializer.learn_one(mini_footprints)
        result = initializer.transform_one()

        dense_matrix = result.data.todense()
        assert np.allclose(dense_matrix, dense_matrix.T)

    def test_diagonal(
        self, initializer: OverlapsInitializer, mini_footprints: xr.DataArray
    ) -> None:
        """Test that diagonal elements are properly set."""
        initializer.learn_one(mini_footprints)
        result = initializer.transform_one()

        dense_matrix = result.data.todense()
        assert np.allclose(np.diag(dense_matrix), 1)
