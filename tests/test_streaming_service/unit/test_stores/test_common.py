import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import ObservableStore, Component
from cala.streaming.stores.common import FootprintStore, TraceStore


class TestFootprints:
    """Test suite for the Footprints class."""

    @pytest.fixture
    def sample_footprints(self):
        """Create sample footprint data."""
        data = np.random.rand(3, 10, 10)  # 3 components, 10x10 spatial dimensions
        coords = {
            "id_": ("components", ["id0", "id1", "id2"]),
            "type_": (
                "components",
                [Component.NEURON, Component.NEURON, Component.BACKGROUND],
            ),
        }
        return FootprintStore(
            xr.DataArray(data, dims=("components", "height", "width"), coords=coords)
        )

    def test_initialization(self, sample_footprints):
        """Test proper initialization of Footprints."""
        assert isinstance(sample_footprints, ObservableStore)
        assert isinstance(sample_footprints, FootprintStore)
        assert sample_footprints.warehouse.dims == ("components", "height", "width")
        assert "id_" in sample_footprints.warehouse.coords
        assert "type_" in sample_footprints.warehouse.coords


class TestTraces:
    """Test suite for the Traces class."""

    @pytest.fixture
    def sample_traces(self):
        """Create sample temporal traces data."""
        data = np.random.rand(3, 100)  # 3 components, 100 timepoints
        coords = {
            "id_": ("components", ["id0", "id1", "id2"]),
            "type_": (
                "components",
                [Component.NEURON, Component.NEURON, Component.BACKGROUND],
            ),
        }
        return TraceStore(
            xr.DataArray(data, dims=("components", "frames"), coords=coords)
        )

    def test_initialization(self, sample_traces):
        """Test proper initialization of Traces."""
        assert isinstance(sample_traces, ObservableStore)
        assert isinstance(sample_traces, TraceStore)
        assert sample_traces.warehouse.dims == ("components", "frames")
        assert "id_" in sample_traces.warehouse.coords
        assert "type_" in sample_traces.warehouse.coords
