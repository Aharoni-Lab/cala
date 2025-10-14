import os
from datetime import datetime
from pathlib import Path

import pytest
import xarray as xr

from cala.assets import Buffer, Traces
from cala.models import AXIS


@pytest.fixture
def path() -> Path:
    return Path("assets")


def test_assign_zarr(path, connected_cells):
    zarr_traces = Traces(zarr_path=path, peek_size=100)
    traces = connected_cells.traces.array
    zarr_traces.array = traces
    print(os.listdir(zarr_traces.zarr_path))
    assert zarr_traces.array_ is None  # not in memory
    assert zarr_traces.array.equals(traces)


def test_from_array(connected_cells, path):
    traces = connected_cells.traces.array
    zarr_traces = Traces.from_array(traces, path, peek_size=connected_cells.n_frames)
    assert zarr_traces.array_ is None
    assert zarr_traces.array.equals(traces)


@pytest.mark.parametrize("peek_shift", [-1, 0, 1])
def test_peek(connected_cells, path, peek_shift):
    traces = connected_cells.traces.array
    zarr_traces = Traces.from_array(traces, path, peek_size=connected_cells.n_frames + peek_shift)
    if peek_shift >= 0:
        assert zarr_traces.array.equals(traces)
    else:
        with pytest.raises(AssertionError):
            assert zarr_traces.array.equals(traces)


def test_ingest_frame(path, connected_cells):
    traces = connected_cells.traces.array
    old_traces = traces.isel({AXIS.frames_dim: slice(None, -1)})
    zarr_traces = Traces.from_array(old_traces, path, peek_size=connected_cells.n_frames)
    new_traces = connected_cells.traces.array.isel({AXIS.frames_dim: [-1]})

    zarr_traces.update(new_traces, append_dim=AXIS.frames_dim)
    # new_traces.to_zarr(zarr_traces.zarr_path, append_dim=AXIS.frames_dim)

    assert zarr_traces.array.equals(traces)


def test_ingest_component(connected_cells, path):
    traces = connected_cells.traces.array
    old_traces = traces.isel({AXIS.component_dim: slice(None, -1)})
    zarr_traces = Traces.from_array(old_traces, path, peek_size=connected_cells.n_frames)
    new_traces = connected_cells.traces.array.isel({AXIS.component_dim: [-1]})

    zarr_traces.update(new_traces, append_dim=AXIS.component_dim)
    # new_traces.to_zarr(zarr_traces.zarr_path, append_dim=AXIS.component_dim)

    assert zarr_traces.array.equals(traces)


def test_overwrite(connected_cells, separate_cells, path):
    conn_traces = connected_cells.traces.array
    zarr_traces = Traces.from_array(conn_traces, path, peek_size=connected_cells.n_frames)

    sep_traces = separate_cells.traces.array
    zarr_traces.array = sep_traces
    assert zarr_traces.array.equals(sep_traces)


# two cases of init:
# 1. brick by brick
# 2. lump dump

# two cases of update
# 1. append
# 2. lump update


def test_buffer_assign(connected_cells):
    movie = connected_cells.make_movie().array
    buff = Buffer(size=10)
    buff.array = movie.isel({AXIS.frames_dim: -1})
    assert buff.array.equals(movie.isel({AXIS.frames_dim: [-1]}))

    buff.array = movie.isel({AXIS.frames_dim: slice(-5, None)})
    assert buff.array.equals(movie.isel({AXIS.frames_dim: slice(-5, None)}))

    buff.array = movie.isel({AXIS.frames_dim: slice(-10, None)})
    assert buff.array.equals(movie.isel({AXIS.frames_dim: slice(-10, None)}))

    buff.array = movie.isel({AXIS.frames_dim: slice(-15, None)})
    assert buff.array.equals(movie.isel({AXIS.frames_dim: slice(-10, None)}))


def test_buffer_append(connected_cells):
    movie = connected_cells.make_movie().array
    buff = Buffer(size=10)
    buff.array = movie.isel({AXIS.frames_dim: 0})
    buff.append(movie.isel({AXIS.frames_dim: 1}))
    assert buff.array.equals(movie.isel({AXIS.frames_dim: slice(0, 2)}))

    buff.array = movie.isel({AXIS.frames_dim: slice(None, 9)})
    buff.append(movie.isel({AXIS.frames_dim: 9}))
    assert buff.array.equals(movie.isel({AXIS.frames_dim: slice(0, 10)}))
    buff.append(movie.isel({AXIS.frames_dim: 10}))
    assert buff.array.equals(movie.isel({AXIS.frames_dim: slice(1, 11)}))


def test_buffer_speed(single_cell):
    movie = single_cell.make_movie().array
    movie = xr.concat([movie, movie], dim=AXIS.frames_dim)
    buff = Buffer(size=100)
    buff.array = movie

    start = datetime.now()
    iter = 100
    for _ in range(iter):
        buff.append(movie.isel({AXIS.frames_dim: 0}))
        buff.array
    result = (datetime.now() - start) / iter

    start = datetime.now()
    for _ in range(iter):
        xr.concat(
            [movie.isel({AXIS.frames_dim: slice(1, None)}), movie.isel({AXIS.frames_dim: 0})],
            dim=AXIS.frames_dim,
        )
    expected = (datetime.now() - start) / iter

    assert result < expected
