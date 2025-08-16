import numpy as np
import pytest
from noob.node import Node, NodeSpecification

from cala.assets import Frame, Overlaps, Traces
from cala.models import AXIS


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

    xray = Node.from_specification(
        spec=NodeSpecification(id="test", type="cala.nodes.overlap.initialize")
    )

    traces = Traces.from_array(toy.traces.array.isel({AXIS.frames_dim: slice(None, -1)}))

    frame = Frame.from_array(toy.make_movie().array.isel({AXIS.frames_dim: -1}))
    overlap = xray.process(overlaps=Overlaps(), footprints=toy.footprints)

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

    new_traces = Traces.from_array(
        toy.traces.array.isel({AXIS.component_dim: [-1], AXIS.frames_dim: slice(None, -10)})
    )

    new_traces.array.attrs["replaces"] = ["cell_0"]

    result = comp_update.process(traces, new_traces)

    expected = toy.traces.array.drop_sel({AXIS.component_dim: 0})
    expected.loc[{AXIS.component_dim: -1, AXIS.frames_dim: slice(40, None)}] = 0

    assert result.array.equals(expected)
