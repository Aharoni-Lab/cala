import numpy as np
import pytest
from noob.node import Node, NodeSpecification

from cala.assets import Residual
from cala.models.axis import AXIS
from cala.nodes.residual import _clear_overestimates
from cala.testing.toy import FrameDims, Position, Toy


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


@pytest.fixture
def one_cell() -> Toy:
    n_frames = 50

    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=3,
        cell_positions=[Position(width=25, height=25)],
        cell_traces=[np.array(range(n_frames), dtype=float)],
    )


def test_clear_overestimates(one_cell) -> None:
    residual = Residual.from_array(one_cell.make_movie().array)
    residual.array.loc[{AXIS.width_coord: slice(one_cell.cell_positions[0].width, None)}] *= -1

    result = _clear_overestimates(A=one_cell.footprints.array, R=residual.array, clip_val=-1.0)
    expected = one_cell.footprints.array.copy()
    expected.loc[{AXIS.width_coord: slice(one_cell.cell_positions[0].width, None)}] = 0

    assert result.equals(expected)
