import numpy as np
import pytest
from noob.node import Node, NodeSpecification

from cala.assets import Frame, Movie, PopSnap, Traces
from cala.models import AXIS
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
            np.zeros(n_frames, dtype=float),
            np.ones(n_frames, dtype=float),
            np.array(range(n_frames), dtype=float),
            np.array(range(n_frames - 1, -1, -1), dtype=float),
        ],
    )


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
        new_traces=Traces.from_array(separate_cells.traces.array.isel({AXIS.component_dim: [-1]})),
    )

    expected = init.process(traces=separate_cells.traces, frames=separate_cells.make_movie())

    assert expected == result
