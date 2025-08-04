import numpy as np
import pytest
from noob.node import NodeSpecification, Node

from cala.assets import Frame, Trace, Traces
from cala.models import AXIS
from cala.nodes.overlap import Overlapper
from cala.nodes.traces import Init, FrameUpdate, ingest_component
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture
def separate_cells() -> Toy:
    n_frames = 50

    return Toy(
        n_frames=n_frames,
        frame_dims=FrameDims(width=50, height=50),
        cell_radii=3,
        cell_positions=[Position(width=15, height=15), Position(width=35, height=35)],
        cell_traces=[
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames, 0, -1), dtype=float),
        ],
    )


@pytest.fixture
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="init_test", type="cala.nodes.traces.Init")
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_init(init, toy, request) -> None:
    toy = request.getfixturevalue(toy)

    traces = init.process(footprints=toy.footprints, frames=toy.make_movie())

    np.testing.assert_array_equal(traces.array, toy.traces.array)


@pytest.fixture
def frame_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(
            id="frame_test", type="cala.nodes.traces.FrameUpdate", params={"tolerance": 1e-3}
        )
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_ingest_frame(frame_update, toy, request) -> None:
    toy = request.getfixturevalue(toy)

    xray = Overlapper.from_specification(
        spec=NodeSpecification(id="test", type="cala.nodes.overlap.Overlapper")
    )

    traces = Traces.from_array(toy.traces.array.isel({AXIS.frames_dim: slice(None, -1)}))

    frame = Frame.from_array(toy.make_movie().array.isel({AXIS.frames_dim: -1}))
    overlap = xray.initialize(footprints=toy.footprints)

    result = frame_update.process(
        traces=traces,
        footprints=toy.footprints,
        frame=frame,
        overlaps=overlap,
    )

    assert result.array.equals(toy.traces.array.isel({AXIS.frames_dim: -1}))


@pytest.fixture
def comp_update() -> Node:
    return Node.from_specification(
        NodeSpecification(id="comp_test", type="cala.nodes.traces.ingest_component")
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_ingest_component(comp_update, toy, request) -> None:
    toy = request.getfixturevalue(toy)

    traces = Traces.from_array(toy.traces.array.isel({AXIS.component_dim: slice(None, -1)}))

    trace = Trace.from_array(toy.traces.array.isel({AXIS.component_dim: -1}))

    expected = comp_update.process(traces, trace)

    assert expected == toy.traces
