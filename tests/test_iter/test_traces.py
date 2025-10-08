import numpy as np
import pytest
import xarray as xr
from noob.node import Node, NodeSpecification

from cala.assets import Frame, Overlaps, Traces
from cala.models import AXIS


@pytest.fixture
def frame_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(
            id="frame_test",
            type="cala.nodes.traces.FrameUpdate",
            params={"max_iter": 100, "tol": 1e-4},
        )
    )


@pytest.mark.parametrize("toy", ["separate_cells", "connected_cells"])
def test_update_traces(frame_update, toy, request) -> None:
    toy = request.getfixturevalue(toy)

    xray = Node.from_specification(
        spec=NodeSpecification(id="test", type="cala.nodes.overlap.initialize")
    )

    traces = Traces.from_array(toy.traces.array.isel({AXIS.frames_dim: slice(None, -1)}))

    frame = Frame.from_array(toy.make_movie().array.isel({AXIS.frames_dim: -1}))
    overlap = xray.process(overlaps=Overlaps(), footprints=toy.footprints)

    result = frame_update.process(
        traces=traces, footprints=toy.footprints, frame=frame, overlaps=overlap
    ).array
    expected = toy.traces.array.isel({AXIS.frames_dim: -1})

    xr.testing.assert_allclose(result, expected, atol=1e-3)


@pytest.fixture
def comp_update() -> Node:
    return Node.from_specification(
        NodeSpecification(id="comp_test", type="cala.nodes.traces.ingest_component")
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_ingest_component(comp_update, toy, request) -> None:
    toy = request.getfixturevalue(toy)

    traces = Traces.from_array(toy.traces.array.isel({AXIS.component_dim: slice(None, -1)}))

    new_traces = toy.traces.array.isel(
        {AXIS.component_dim: [-1], AXIS.frames_dim: slice(-40, None)}
    )

    new_traces.attrs["replaces"] = ["cell_0"]

    result = comp_update.process(traces, Traces.from_array(new_traces))

    expected = toy.traces.array.drop_sel({AXIS.component_dim: 0})
    expected.loc[{AXIS.component_dim: -1, AXIS.frames_dim: slice(None, 10 - 1)}] = np.nan

    assert result.array.equals(expected)
