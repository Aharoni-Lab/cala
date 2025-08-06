import numpy as np
import pytest
from noob.node import Node, NodeSpecification

from cala.assets import Frame, Movie, PopSnap, Traces
from cala.models import AXIS


@pytest.fixture(scope="function")
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ps_init_test", type="cala.nodes.pixel_stats.initialize")
    )


def test_init(init, separate_cells) -> None:
    result = init.process(traces=separate_cells.traces, frames=separate_cells.make_movie())

    movie = separate_cells.make_movie()

    for id_, trace in zip(
        separate_cells.cell_ids, separate_cells.traces.array.transpose(AXIS.component_dim, ...)
    ):
        assert np.all(
            result.array.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: id_})
            == (trace @ movie.array) / separate_cells.n_frames
        )


@pytest.fixture(scope="function")
def frame_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ps_frame_test", type="cala.nodes.pixel_stats.ingest_frame")
    )


def test_ingest_frame(init, frame_update, separate_cells) -> None:

    pre_ingest = init.process(
        traces=Traces.from_array(
            separate_cells.traces.array.isel({AXIS.frames_dim: slice(None, -1)})
        ),
        frames=Movie.from_array(
            separate_cells.make_movie().array.isel({AXIS.frames_dim: slice(None, -1)})
        ),
    )

    result = frame_update.process(
        pixel_stats=pre_ingest,
        frame=Frame.from_array(separate_cells.make_movie().array.isel({AXIS.frames_dim: -1})),
        new_traces=PopSnap.from_array(separate_cells.traces.array.isel({AXIS.frames_dim: -1})),
    )

    expected = init.process(traces=separate_cells.traces, frames=separate_cells.make_movie())
    assert expected == result


@pytest.fixture(scope="function")
def comp_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ps_comp_test", type="cala.nodes.pixel_stats.ingest_component")
    )


def test_ingest_component(init, comp_update, separate_cells):
    pre_ingest = init.process(
        traces=Traces.from_array(
            separate_cells.traces.array.isel({AXIS.component_dim: slice(None, -1)})
        ),
        frames=separate_cells.make_movie(),
    )

    result = comp_update.process(
        pixel_stats=pre_ingest,
        frames=separate_cells.make_movie(),
        new_trace=Traces.from_array(separate_cells.traces.array.isel({AXIS.component_dim: [-1]})),
        traces=separate_cells.traces,
    )

    expected = init.process(traces=separate_cells.traces, frames=separate_cells.make_movie())

    assert expected == result
