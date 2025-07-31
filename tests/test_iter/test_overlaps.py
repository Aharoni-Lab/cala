import numpy as np
import pytest
from noob.node import NodeSpecification

from cala.assets import Footprints
from cala.models import AXIS
from cala.nodes.overlap import Overlapper
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture(scope="function")
def overlapper() -> Overlapper:
    return Overlapper.from_specification(
        spec=NodeSpecification(id="overlap_test", type="cala.nodes.overlap.Overlapper")
    )


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
            np.zeros(n_frames, dtype=float),
            np.ones(n_frames, dtype=float),
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames - 1, -1, -1), dtype=float),
        ],
    )


def test_init(overlapper, separate_cells, connected_cells) -> None:
    overlap = overlapper.initialize(footprints=separate_cells.footprints)

    assert np.trace(overlap.array) == len(separate_cells.cell_ids)
    assert np.all(np.triu(overlap.array, k=1) == 0)

    result = overlapper.initialize(footprints=connected_cells.footprints)

    expected = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])

    np.testing.assert_array_equal(result.array, expected)


@pytest.mark.parametrize("toy", ["separate_cells", "connected_cells"])
def test_ingest_component(overlapper, toy, request) -> None:
    toy = request.getfixturevalue(toy)
    base = Footprints.from_array(toy.footprints.array.isel({AXIS.component_dim: slice(None, -1)}))
    new = Footprints.from_array(toy.footprints.array.isel({AXIS.component_dim: [-1]}))

    overlapper.initialize(footprints=base)

    result = overlapper.ingest_component(footprints=base, new_footprints=new)
    expected = overlapper.initialize(footprints=toy.footprints)

    assert result == expected
