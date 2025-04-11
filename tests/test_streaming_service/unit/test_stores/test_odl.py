import numpy as np
import pytest
import sparse
import xarray as xr

from cala.streaming.core import Component, ObservableStore
from cala.streaming.stores.odl import (
    ComponentStatStore,
    OverlapStore,
    PixelStatStore,
    ResidualStore,
)


class TestPixelStats:
    """Test suite for PixelStats store."""

    @pytest.fixture
    def sample_pixel_stats(self) -> PixelStatStore:
        """Create sample pixel statistics data."""
        n_pixels = 100  # 10x10 image
        n_components = 3
        data = np.random.rand(n_pixels, n_components)
        coords = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": (
                "components",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
        }
        pixelstore = PixelStatStore()
        pixelstore.warehouse = xr.DataArray(data, dims=("pixels", "components"), coords=coords)
        return pixelstore

    def test_initialization(self, sample_pixel_stats: PixelStatStore) -> None:
        """Test proper initialization of PixelStats."""
        assert isinstance(sample_pixel_stats, ObservableStore)
        assert sample_pixel_stats.warehouse.dims == ("pixels", "components")
        assert "id_" in sample_pixel_stats.warehouse.coords
        assert "type_" in sample_pixel_stats.warehouse.coords

    def test_data_consistency(self, sample_pixel_stats: PixelStatStore) -> None:
        """Test data and coordinate consistency."""
        assert sample_pixel_stats.warehouse.shape[1] == 3  # number of components
        assert len(sample_pixel_stats.warehouse.coords["id_"]) == 3
        assert sample_pixel_stats.warehouse.coords["type_"].values.tolist() == [
            Component.NEURON.value,
            Component.NEURON.value,
            Component.BACKGROUND.value,
        ]


class TestComponentStats:
    """Test suite for ComponentStats store."""

    @pytest.fixture
    def sample_component_stats(self) -> ComponentStatStore:
        """Create sample component correlation matrix."""
        n_components = 3
        data = np.random.rand(n_components, n_components)
        # Make it symmetric as correlation matrices should be
        data = (data + data.T) / 2
        np.fill_diagonal(data, 1.0)  # Diagonal should be 1s

        coords = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": (
                "components",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
        }
        compstore = ComponentStatStore()
        compstore.warehouse = xr.DataArray(data, dims=("components", "components"), coords=coords)

        return compstore

    def test_initialization(self, sample_component_stats: ComponentStatStore) -> None:
        """Test proper initialization of ComponentStats."""
        assert isinstance(sample_component_stats, ComponentStatStore)
        assert sample_component_stats.warehouse.dims == ("components", "components")
        assert "id_" in sample_component_stats.warehouse.coords
        assert "type_" in sample_component_stats.warehouse.coords


class TestResidual:
    """Test suite for Residual store."""

    @pytest.fixture
    def sample_residual(self) -> ResidualStore:
        """Create sample residual data."""
        height, width = 10, 10
        n_frames = 5
        data = np.random.randn(height, width, n_frames)  # Should be zero-centered
        residstore = ResidualStore()
        residstore.warehouse = xr.DataArray(data, dims=("height", "width", "frames"))
        return residstore

    def test_initialization(self, sample_residual: ResidualStore) -> None:
        """Test proper initialization of Residual."""
        assert isinstance(sample_residual, ResidualStore)
        assert sample_residual.warehouse.dims == ("height", "width", "frames")


class TestOverlapGroups:
    """Test suite for OverlapGroups store."""

    @pytest.fixture
    def sample_overlap_groups(self) -> OverlapStore:
        """Create sample overlap groups using sparse matrix."""
        n_components = 5

        # Create sparse matrix with some overlapping components
        coords = ([0, 1, 1, 2], [1, 0, 2, 1])  # Example overlap pattern
        data = np.ones(len(coords[0]))

        sparse_matrix = sparse.COO(coords=coords, data=data, shape=(n_components, n_components))

        coords_dict = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": (
                "components",
                [Component.NEURON] * 3 + [Component.BACKGROUND] * 2,
            ),
        }
        overlapstore = OverlapStore()
        overlapstore.warehouse = xr.DataArray(
            sparse_matrix, dims=("components", "components"), coords=coords_dict
        )

        return overlapstore

    def test_initialization(self, sample_overlap_groups: OverlapStore) -> None:
        """Test proper initialization of OverlapGroups."""
        assert isinstance(sample_overlap_groups, OverlapStore)
        assert sample_overlap_groups.warehouse.dims == ("components", "components")
        assert "id_" in sample_overlap_groups.warehouse.coords
        assert "type_" in sample_overlap_groups.warehouse.coords
