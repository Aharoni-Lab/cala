import numpy as np
import pytest
from noob.node import NodeSpecification, Node

from cala.assets import Frame, Movie, PopSnap, Traces
from cala.models import AXIS
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture
def separate_cells() -> Toy:
    n_frames = 50

    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=3,
        cell_positions=[
            Position(width=15, height=15),
            Position(width=15, height=35),
            Position(width=25, height=25),
            Position(width=35, height=35),
        ],
        cell_traces=[
            np.zeros(n_frames, dtype=float),
            np.ones(n_frames, dtype=float),
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames - 1, -1, -1), dtype=float),
        ],
    )


@pytest.fixture(scope="function")
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="res_init_test", type="cala.nodes.residual.initialize")
    )


def test_init(init, separate_cells) -> None:
    result = init.process(
        footprints=separate_cells.footprints,
        traces=separate_cells.traces,
        frames=separate_cells.make_movie(),
    )

    assert np.all(result.array == 0)


@pytest.fixture(scope="function")
def frame_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="res_frame_test", type="cala.nodes.residual.ingest_frame")
    )


def test_ingest_frame(init, frame_update, separate_cells) -> None:

    pre_ingest = init.process(
        footprints=separate_cells.footprints,
        traces=Traces.from_array(
            separate_cells.traces.array.isel({AXIS.frames_dim: slice(None, -1)})
        ),
        frames=Movie.from_array(
            separate_cells.make_movie().array.isel({AXIS.frames_dim: slice(None, -1)})
        ),
    )

    residual = frame_update.process(
        residual=pre_ingest,
        frame=Frame.from_array(separate_cells.make_movie().array.isel({AXIS.frames_dim: -1})),
        footprints=separate_cells.footprints,
        traces=PopSnap.from_array(separate_cells.traces.array.isel({AXIS.frames_dim: -1})),
    )

    assert np.all(residual.array == 0)
