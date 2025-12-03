import numpy as np
import pytest
import xarray as xr
from noob.node import Node, NodeSpecification

from cala.assets import AXIS
from cala.assets.assets import CompStats, Footprints, PixStats
from cala.nodes.omf.footprints import ingest_component
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture
def four_separate_cells() -> Toy:
    n_frames = 50

    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=1,
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
    )


@pytest.fixture
def fpter() -> Node:
    return Node.from_specification(
        NodeSpecification(
            id="test_footprinter",
            type="cala.nodes.omf.footprints.Footprinter",
            params={"max_iter": 5, "ratio_lb": 0.10},
        )
    )


@pytest.mark.parametrize("toy", ["four_separate_cells", "four_connected_cells"])
def test_ingest_frame(fpter, toy, request):
    toy = request.getfixturevalue(toy)

    pixstats = Node.from_specification(
        NodeSpecification(id="test_pixstats", type="cala.nodes.omf.pixel_stats.initialize")
    ).process(
        traces=toy.traces.array, frames=toy.make_movie().array, footprints=toy.footprints.array
    )
    compstats = Node.from_specification(
        NodeSpecification(id="test_compstats", type="cala.nodes.omf.component_stats.initialize")
    ).process(traces=toy.traces.array)

    result = fpter.process(
        footprints=toy.footprints,
        pixel_stats=PixStats.from_array(pixstats),
        component_stats=CompStats.from_array(compstats),
    ).array.as_numpy()

    expected = toy.footprints.array.as_numpy()

    xr.testing.assert_allclose(result, expected)


@pytest.mark.parametrize("toy", ["four_separate_cells"])
def test_ingest_component(toy, request) -> None:
    toy = request.getfixturevalue(toy)

    footprints = toy.footprints.array.isel({AXIS.component_dim: slice(None, -1)})

    new_footprints = toy.footprints.array.isel({AXIS.component_dim: [-1]})

    new_footprints.attrs["replaces"] = ["cell_0"]

    result = ingest_component(
        Footprints.from_array(footprints), Footprints.from_array(new_footprints)
    )

    expected = toy.footprints.array.drop_sel({AXIS.component_dim: 0})

    assert result.array.equals(expected)
