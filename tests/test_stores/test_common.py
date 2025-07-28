import numpy as np
import pytest
import xarray as xr

from cala.models import AXIS
from cala.stores.common import TraceStore


@pytest.fixture
def trace_store(tmp_path) -> TraceStore:
    """Create sample temporal traces data."""
    data = np.random.rand(3, 100)  # 3 components, 100 timepoints
    peek_size = 1000
    coords = {
        AXIS.id_coord: (AXIS.component_dim, ["id0", "id1", "id2"]),
        AXIS.frame_coord: (AXIS.frames_dim, [i for i in range(100)]),
        AXIS.timestamp_coord: (AXIS.frames_dim, [f"0{i}" for i in range(100)]),
    }
    trace_store = TraceStore(peek_size=peek_size, store_dir=tmp_path)
    trace_store.warehouse = xr.DataArray(
        data, dims=(AXIS.component_dim, AXIS.frames_dim), coords=coords
    )
    return trace_store


def test_append_new_frames(trace_store):
    """Test appending new frames to existing components."""
    store = trace_store

    # Create new frame data for existing components
    new_data = np.random.rand(3, 2)  # 2 new frames
    coords = {
        AXIS.id_coord: (AXIS.component_dim, ["id0", "id1", "id2"]),
        AXIS.frame_coord: (AXIS.frames_dim, [105, 106]),  # New frame indices
        AXIS.timestamp_coord: (AXIS.frames_dim, ["105", "106"]),  # New frame indices
    }
    new_frames = xr.DataArray(new_data, dims=(AXIS.component_dim, AXIS.frames_dim), coords=coords)

    store._append(new_frames, append_dim=AXIS.frames_dim)

    # Verify the append
    result = store.warehouse
    assert result.sizes == {
        "component": 3,
        AXIS.frames_dim: 102,
    }  # 3 components, 102 frames total


def test_append_new_components(trace_store):
    """Test appending new components."""
    store = trace_store

    # Create new component data
    new_data = np.random.rand(2, 100)  # 2 new components, 5 frames
    coords = {
        AXIS.id_coord: (AXIS.component_dim, ["id3", "id4"]),
        AXIS.frame_coord: (AXIS.frames_dim, [i for i in range(100)]),
        AXIS.timestamp_coord: (AXIS.frames_dim, [f"0{i}" for i in range(100)]),
    }
    new_components = xr.DataArray(
        new_data, dims=(AXIS.component_dim, AXIS.frames_dim), coords=coords
    )

    store._append(new_components, append_dim=AXIS.component_dim)

    # Verify the append
    result = store.warehouse
    assert result.sizes == {
        AXIS.component_dim: 5,
        AXIS.frames_dim: 100,
    }  # 5 components total, 100 frames


def test_update_existing_components(trace_store):
    """Test updating traces for existing components with new frames."""
    store = trace_store

    # Create update data for existing components
    new_data = np.ones((3, 10))  # 10 new frames
    coords = {
        AXIS.id_coord: (AXIS.component_dim, ["id0", "id1", "id2"]),
        AXIS.frame_coord: (AXIS.frames_dim, [i for i in range(100, 110)]),
        AXIS.timestamp_coord: (AXIS.frames_dim, [f"0{i}" for i in range(100, 110)]),
    }
    update_data = xr.DataArray(new_data, dims=(AXIS.component_dim, AXIS.frames_dim), coords=coords)

    store.update(update_data)

    # Verify the update
    result = store.warehouse
    assert result.sizes == {AXIS.component_dim: 3, AXIS.frames_dim: 110}


def test_update_new_components(trace_store):
    """Test updating with new components (including backfill)."""
    store = trace_store

    # Create new component data with fewer frames (needs backfill)
    new_data = np.random.rand(2, 10)  # 2 new components, 10 buffer frames
    coords = {
        AXIS.id_coord: (AXIS.component_dim, ["id3", "id4"]),
        AXIS.frame_coord: (AXIS.frames_dim, [i for i in range(90, 100)]),
        AXIS.timestamp_coord: (AXIS.frames_dim, [f"0{i}" for i in range(90, 100)]),
    }
    new_components = xr.DataArray(
        new_data, dims=(AXIS.component_dim, AXIS.frames_dim), coords=coords
    )

    store.update(new_components)

    # Verify the update with backfill
    result = store.warehouse
    assert result.shape == (5, 100)  # 5 components total, 5 frames
    # Check that backfilled values are zero
    new_components_data = result.set_xindex(AXIS.id_coord).sel(id_=["id3", "id4"])
    assert np.all(new_components_data.isel(frame=slice(0, 90)) == 0)


def test_update_empty_data(trace_store):
    """Test updating with empty data."""
    store = trace_store

    # Create empty data
    empty_data = xr.DataArray(
        np.array([]).reshape(0, 0),
        dims=(AXIS.component_dim, AXIS.frames_dim),
        coords={
            AXIS.id_coord: (AXIS.component_dim, []),
            AXIS.frame_coord: (AXIS.frames_dim, []),
        },
    )

    store.update(empty_data)

    # Verify no changes
    result = store.warehouse
    assert result.shape == (3, 100)  # Original shape maintained
