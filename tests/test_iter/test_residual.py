import numpy as np
import pytest
from noob.node import Node, NodeSpecification

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
        spec=NodeSpecification(id="res_init_test", type="cala.nodes.residual.build")
    )


def test_init(init, separate_cells) -> None:
    result = init.process(
        footprints=separate_cells.footprints,
        traces=separate_cells.traces,
        frames=separate_cells.make_movie(),
        trigger=True,
    )

    assert np.all(result.array == 0)
