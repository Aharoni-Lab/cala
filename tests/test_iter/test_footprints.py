import numpy as np
import pytest
from noob.node import Node, NodeSpecification

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
            np.random.randint(low=0, high=n_frames, size=n_frames).astype(float),
            np.ones(n_frames, dtype=float),
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames - 1, -1, -1), dtype=float),
        ],
    )


@pytest.fixture
def fpter() -> Node:
    return Node.from_specification(
        NodeSpecification(
            id="test_footprinter",
            type="cala.nodes.footprints.Footprinter",
            params={"boundary_expansion_pixels": None, "tolerance": 1e-7},
        )
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_ingest_frame(fpter, toy, request):
    toy = request.getfixturevalue(toy)

    pixstats = Node.from_specification(
        NodeSpecification(id="test_pixstats", type="cala.nodes.pixel_stats.initialize")
    ).process(traces=toy.traces, frames=toy.make_movie())
    compstats = Node.from_specification(
        NodeSpecification(id="test_compstats", type="cala.nodes.component_stats.initialize")
    ).process(traces=toy.traces)

    result = fpter.process(
        footprints=toy.footprints, pixel_stats=pixstats, component_stats=compstats
    )

    expected = toy.footprints.copy()

    assert result == expected
