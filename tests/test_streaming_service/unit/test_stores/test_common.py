from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import Component, ObservableStore
from cala.streaming.stores.common import FootprintStore, TraceStore


class TestFootprints:
    """Test suite for the Footprints class."""

    @pytest.fixture
    def sample_footprints(self) -> FootprintStore:
        """Create sample footprint data."""
        data = np.random.rand(3, 10, 10)  # 3 components, 10x10 spatial dimensions
        coords = {
            "id_": ("components", ["id0", "id1", "id2"]),
            "type_": (
                "components",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
        }
        footprintstore = FootprintStore()
        footprintstore.warehouse = xr.DataArray(
            data, dims=("components", "height", "width"), coords=coords
        )

        return footprintstore

    def test_initialization(self, sample_footprints: FootprintStore) -> None:
        """Test proper initialization of Footprints."""
        assert isinstance(sample_footprints, ObservableStore)
        assert isinstance(sample_footprints, FootprintStore)
        assert sample_footprints.warehouse.dims == ("components", "height", "width")
        assert "id_" in sample_footprints.warehouse.coords
        assert "type_" in sample_footprints.warehouse.coords


class TestTraces:
    """Test suite for the Traces class."""

    @pytest.fixture
    def sample_traces(self, tmp_path) -> tuple[TraceStore, Path]:
        """Create sample temporal traces data."""
        data = np.random.rand(3, 100)  # 3 components, 100 timepoints
        peek_size = 1000
        coords = {
            "id_": ("component", ["id0", "id1", "id2"]),
            "type_": (
                "component",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
            "frame_": ("frame", [i for i in range(100)]),
            "time_": ("frame", [f"0{i}" for i in range(100)]),
        }
        trace_store = TraceStore(peek_size=peek_size, store_dir=tmp_path)
        trace_store.warehouse = xr.DataArray(data, dims=("component", "frame"), coords=coords)
        return trace_store, tmp_path

    def test_initialization(self, sample_traces: xr.DataArray) -> None:
        """Test proper initialization of Traces."""
        store, path = sample_traces
        assert store.warehouse.dims == ("component", "frame")
        assert "id_" in store.warehouse.coords
        assert "type_" in store.warehouse.coords

    def test_append_new_frames(self, sample_traces):
        """Test appending new frames to existing components."""
        store, path = sample_traces

        # Create new frame data for existing components
        new_data = np.random.rand(3, 2)  # 2 new frames
        coords = {
            "id_": ("component", ["id0", "id1", "id2"]),
            "type_": (
                "component",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
            "frame_": ("frame", [105, 106]),  # New frame indices
            "time_": ("frame", ["105", "106"]),  # New frame indices
        }
        new_frames = xr.DataArray(new_data, dims=("component", "frame"), coords=coords)

        store._append(new_frames, append_dim="frame")

        # Verify the append
        result = store.warehouse
        assert result.sizes == {"component": 3, "frame": 102}  # 3 components, 102 frames total

    def test_append_new_components(self, sample_traces):
        """Test appending new components."""
        store, path = sample_traces

        # Create new component data
        new_data = np.random.rand(2, 100)  # 2 new components, 5 frames
        coords = {
            "id_": ("component", ["id3", "id4"]),
            "type_": (
                "component",
                [Component.NEURON.value, Component.BACKGROUND.value],
            ),
            "frame_": ("frame", [i for i in range(100)]),
            "time_": ("frame", [f"0{i}" for i in range(100)]),
        }
        new_components = xr.DataArray(new_data, dims=("component", "frame"), coords=coords)

        store._append(new_components, append_dim="component")

        # Verify the append
        result = store.warehouse
        assert result.sizes == {"component": 5, "frame": 100}  # 5 components total, 100 frames

    def test_update_existing_components(self, sample_traces):
        """Test updating traces for existing components with new frames."""
        store, path = sample_traces

        # Create update data for existing components
        new_data = np.ones((3, 10))  # 10 new frames
        coords = {
            "id_": ("component", ["id0", "id1", "id2"]),
            "type_": (
                "component",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
            "frame_": ("frame", [i for i in range(100, 110)]),
            "time_": ("frame", [f"0{i}" for i in range(100, 110)]),
        }
        update_data = xr.DataArray(new_data, dims=("component", "frame"), coords=coords)

        store.update(update_data)

        # Verify the update
        result = store.warehouse
        assert result.sizes == {"component": 3, "frame": 110}

    def test_update_new_components(self, sample_traces):
        """Test updating with new components (including backfill)."""
        store, path = sample_traces

        # Create new component data with fewer frames (needs backfill)
        new_data = np.random.rand(2, 10)  # 2 new components, 10 buffer frames
        coords = {
            "id_": ("component", ["id3", "id4"]),
            "type_": (
                "component",
                [Component.NEURON.value, Component.BACKGROUND.value],
            ),
            "frame_": ("frame", [i for i in range(90, 100)]),
            "time_": ("frame", [f"0{i}" for i in range(90, 100)]),
        }
        new_components = xr.DataArray(new_data, dims=("component", "frame"), coords=coords)

        store.update(new_components)

        # Verify the update with backfill
        result = store.warehouse
        assert result.shape == (5, 100)  # 5 components total, 5 frames
        # Check that backfilled values are zero
        new_components_data = result.set_xindex("id_").sel(id_=["id3", "id4"])
        assert np.all(new_components_data.isel(frame=slice(0, 90)) == 0)

    def test_update_empty_data(self, sample_traces):
        """Test updating with empty data."""
        store, path = sample_traces

        # Create empty data
        empty_data = xr.DataArray(
            np.array([]).reshape(0, 0),
            dims=("component", "frame"),
            coords={"id_": ("component", []), "frame": ("frame", [])},
        )

        store.update(empty_data)

        # Verify no changes
        result = store.warehouse
        assert result.shape == (3, 100)  # Original shape maintained
