import numpy as np
import pytest
from noob.node import NodeSpecification

from cala.nodes.iter.component_stats import CompStater
from cala.nodes.iter.footprints import Footprinter
from cala.nodes.iter.pixel_stats import PixelStater
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture
def fpter() -> Footprinter:
    return Footprinter.from_specification(
        NodeSpecification(id="test-footprinter", type="cala.nodes.iter.footprints.Footprinter")
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
            np.random.randint(low=0, high=n_frames, size=n_frames).astype(float),
            np.ones(n_frames, dtype=float),
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames - 1, -1, -1), dtype=float),
        ],
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_ingest_frame(fpter, toy, request):
    toy = request.getfixturevalue(toy)

    pixstats = PixelStater.from_specification(
        NodeSpecification(id="test-pixstats", type="cala.nodes.iter.pixel_stats.PixelStater")
    ).process(traces=toy.traces, frames=toy.make_movie())
    compstats = CompStater.from_specification(
        NodeSpecification(id="test-compstats", type="cala.nodes.iter.component_stats.CompStater")
    ).process(traces=toy.traces)
    fpter.footprints_ = toy.footprints

    result = fpter.ingest_frame(pixel_stats=pixstats, component_stats=compstats)

    expected = toy.footprints.copy()

    assert result == expected
