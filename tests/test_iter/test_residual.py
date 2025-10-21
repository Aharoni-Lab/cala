import numpy as np
import pytest
import xarray as xr
from noob.node import Node, NodeSpecification

from cala.assets import Buffer, Footprints, Frame, Traces
from cala.models.axis import AXIS
from cala.nodes.residual import _align_overestimates, _find_unlayered_footprints
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture
def connected_cells() -> Toy:
    n_frames = 50

    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=8,
        cell_positions=[
            Position(width=15, height=15),
            Position(width=15, height=35),
            Position(width=25, height=25),
            Position(width=35, height=35),
        ],
        cell_traces=[
            np.random.randint(low=0, high=n_frames, size=n_frames).astype(float),
            np.abs(np.sin(np.linspace(-np.pi, np.pi, n_frames)) * n_frames).astype(float),
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames - 1, -1, -1), dtype=float),
        ],
        detected_ons=[n_frames - 1] * 4,
    )


@pytest.fixture(scope="function")
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(
            id="res_init_test",
            type="cala.nodes.residual.Residuer",
        )
    )


def test_init(init, connected_cells) -> None:
    residual = Buffer(size=100)
    gen = connected_cells.movie_gen()

    for _ in range(connected_cells.n_frames - 1):
        init.process(
            residuals=residual,
            frame=Frame.from_array(next(gen)),
            footprints=Footprints(),
            traces=Traces(),
        )
    result, _ = init.process(
        residuals=residual,
        footprints=connected_cells.footprints,
        traces=connected_cells.traces,
        frame=Frame.from_array(next(gen)),
    )

    assert np.all(result.array == 0)


def test_align_overestimates(single_cell) -> None:
    """
    grab the last frame of the residual. assume part of the footprint masked area is negative
    traces needs to proportionally decrease

    Maybe this can be absorbed straight into trace frame_ingest as a constraint.
    """
    movie = single_cell.make_movie()
    last_frame = movie.array.isel({AXIS.frames_dim: -1})

    last_res = xr.zeros_like(last_frame)
    # we have negative residuals
    last_res.loc[{AXIS.width_coord: slice(single_cell.cell_positions[0].width, None)}] = -1
    last_res = last_res.where(single_cell.footprints.array[0].to_numpy(), 0)

    last_trace = single_cell.traces.array.isel({AXIS.frames_dim: -1})

    footprints = single_cell.footprints.array
    shapes_sparse = footprints.data.reshape((footprints.sizes[AXIS.component_dim], -1)).tocsr()

    adjusted_traces = _align_overestimates(
        A_pix=shapes_sparse, R_latest=last_res, C_latest=last_trace
    )

    # adjusted to lower than last_trace
    assert single_cell.traces.array.isel({AXIS.frames_dim: -2}) < adjusted_traces < last_trace


def test_find_exposed_footprints(connected_cells) -> None:
    footprints = connected_cells.footprints.array
    result = _find_unlayered_footprints(
        footprints.data.reshape((footprints.sizes[AXIS.component_dim], -1))
    )
    assert result.max().item() == footprints.max().item()


@pytest.mark.xfail
def test_handle_outlier_pixel() -> None:
    """a test to make sure an outlier pixel does not mess up the whole trace"""


def test_std(init, connected_cells) -> None:
    residual = Buffer(size=100)
    gen = connected_cells.movie_gen()

    for _ in range(connected_cells.n_frames):
        _, result = init.process(
            residuals=residual,
            frame=Frame.from_array(next(gen)),
            footprints=Footprints(),
            traces=Traces(),
        )

    expected = connected_cells.make_movie().array.std(dim=AXIS.frames_dim).values

    assert np.allclose(result, expected)

    gen = connected_cells.movie_gen()
    _, result = init.process(
        residuals=residual,
        footprints=connected_cells.footprints,
        traces=connected_cells.traces,
        frame=Frame.from_array(next(gen)),
    )

    assert np.all(result == 0)
