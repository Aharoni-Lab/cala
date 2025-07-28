import numpy as np
import pytest
from noob.node import NodeSpecification

from cala.models import AXIS, Frame, Traces, Trace
from cala.nodes.iter.overlap import Overlapper
from cala.nodes.iter.traces import Tracer
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
def tracer() -> Tracer:
    return Tracer.from_specification(
        spec=NodeSpecification(
            id="tracer-test", type="cala.nodes.iter.traces.Tracer", params={"tolerance": 1e-3}
        )
    )


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_init(tracer, toy, request) -> None:
    toy = request.getfixturevalue(toy)

    traces = tracer.initialize(footprints=toy.footprints, frames=toy.make_movie())

    np.testing.assert_array_equal(traces.array, toy.traces.array)


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_ingest_frame(tracer, toy, request) -> None:
    toy = request.getfixturevalue(toy)

    xray = Overlapper.from_specification(
        spec=NodeSpecification(id="test", type="cala.nodes.iter.overlap.Overlapper")
    )

    traces = Traces(array=toy.traces.array.isel({AXIS.frames_dim: slice(None, -1)}))
    tracer.traces_ = traces

    frame = Frame(array=toy.make_movie().array.isel({AXIS.frames_dim: -1}))
    overlap = xray.initialize(footprints=toy.footprints)

    result = tracer.ingest_frame(
        footprints=toy.footprints,
        frame=frame,
        overlaps=overlap,
    )

    assert result.array.equals(toy.traces.array)


@pytest.mark.parametrize("toy", ["separate_cells"])
def test_ingest_component(tracer, toy, request) -> None:
    toy = request.getfixturevalue(toy)

    traces = Traces(array=toy.traces.array.isel({AXIS.component_dim: slice(None, -1)}))
    tracer.traces_ = traces

    trace = Trace(array=toy.traces.array.isel({AXIS.component_dim: -1}))

    expected = tracer.ingest_component(trace)

    assert expected == toy.traces
