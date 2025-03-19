import numpy as np
import pytest
import sparse
import xarray as xr

from cala.streaming.stores.odl import (
    PixelStats,
    ComponentStats,
    Residual,
    OverlapGroups,
)


class TestPixelStats:
    """Test suite for PixelStats store."""

    @pytest.fixture
    def sample_pixel_stats(self):
        """Create sample pixel statistics data."""
        n_pixels = 100  # 10x10 image
        n_components = 3
        data = np.random.rand(n_pixels, n_components)
        coords = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": ("components", ["neuron", "neuron", "background"]),
        }
        return PixelStats(data, dims=("pixels", "components"), coords=coords)

    def test_initialization(self, sample_pixel_stats):
        """Test proper initialization of PixelStats."""
        assert isinstance(sample_pixel_stats, PixelStats)
        assert isinstance(sample_pixel_stats, xr.DataArray)
        assert sample_pixel_stats.dims == ("pixels", "components")
        assert "id_" in sample_pixel_stats.coords
        assert "type_" in sample_pixel_stats.coords

    def test_data_consistency(self, sample_pixel_stats):
        """Test data and coordinate consistency."""
        assert sample_pixel_stats.shape[1] == 3  # number of components
        assert len(sample_pixel_stats.coords["id_"]) == 3
        assert list(sample_pixel_stats.coords["type_"].values) == [
            "neuron",
            "neuron",
            "background",
        ]


class TestComponentStats:
    """Test suite for ComponentStats store."""

    @pytest.fixture
    def sample_component_stats(self):
        """Create sample component correlation matrix."""
        n_components = 3
        data = np.random.rand(n_components, n_components)
        # Make it symmetric as correlation matrices should be
        data = (data + data.T) / 2
        np.fill_diagonal(data, 1.0)  # Diagonal should be 1s

        coords = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": ("components", ["neuron", "neuron", "background"]),
        }
        return ComponentStats(data, dims=("components", "components"), coords=coords)

    def test_initialization(self, sample_component_stats):
        """Test proper initialization of ComponentStats."""
        assert isinstance(sample_component_stats, ComponentStats)
        assert isinstance(sample_component_stats, xr.DataArray)
        assert sample_component_stats.dims == ("components", "components")
        assert "id_" in sample_component_stats.coords
        assert "type_" in sample_component_stats.coords


class TestResidual:
    """Test suite for Residual store."""

    @pytest.fixture
    def sample_residual(self):
        """Create sample residual data."""
        height, width = 10, 10
        n_frames = 5
        data = np.random.randn(height, width, n_frames)  # Should be zero-centered
        return Residual(data, dims=("height", "width", "frames"))

    def test_initialization(self, sample_residual):
        """Test proper initialization of Residual."""
        assert isinstance(sample_residual, Residual)
        assert isinstance(sample_residual, xr.DataArray)
        assert sample_residual.dims == ("height", "width", "frames")


class TestOverlapGroups:
    """Test suite for OverlapGroups store."""

    @pytest.fixture
    def sample_overlap_groups(self):
        """Create sample overlap groups using sparse matrix."""
        n_components = 5

        # Create sparse matrix with some overlapping components
        coords = ([0, 1, 1, 2], [1, 0, 2, 1])  # Example overlap pattern
        data = np.ones(len(coords[0]))

        sparse_matrix = sparse.COO(
            coords=coords, data=data, shape=(n_components, n_components)
        )

        coords_dict = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": ("components", ["neuron"] * 3 + ["background"] * 2),
        }

        return OverlapGroups(
            sparse_matrix, dims=("components", "components"), coords=coords_dict
        )

    def test_initialization(self, sample_overlap_groups):
        """Test proper initialization of OverlapGroups."""
        assert isinstance(sample_overlap_groups, OverlapGroups)
        assert isinstance(sample_overlap_groups, xr.DataArray)
        assert sample_overlap_groups.dims == ("components", "components")
        assert "id_" in sample_overlap_groups.coords
        assert "type_" in sample_overlap_groups.coords
