import numpy as np
import pytest
from noob.node import NodeSpecification

from cala.assets import Frame, Movie, PopSnap, Traces
from cala.models import AXIS
from cala.nodes.iter.pixel_stats import PixelStater
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture(scope="function")
def pix_stats() -> PixelStater:
    return PixelStater.from_specification(
        spec=NodeSpecification(id="pix_stat_test", type="cala.nodes.iter.pixel_stats.PixelStater")
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


def test_init(pix_stats, separate_cells) -> None:
    result = pix_stats.initialize(traces=separate_cells.traces, frames=separate_cells.make_movie())

    movie = separate_cells.make_movie()

    for id_, trace in zip(
        separate_cells.cell_ids, separate_cells.traces.array.transpose(AXIS.component_dim, ...)
    ):
        assert np.all(
            result.array.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: id_})
            == (trace @ movie.array) / separate_cells.n_frames
        )


def test_ingest_frame(pix_stats, separate_cells) -> None:

    pix_stats.initialize(
        traces=Traces.from_array(
            separate_cells.traces.array.isel({AXIS.frames_dim: slice(None, -1)})
        ),
        frames=Movie.from_array(
            separate_cells.make_movie().array.isel({AXIS.frames_dim: slice(None, -1)})
        ),
    )

    result = pix_stats.ingest_frame(
        frame=Frame.from_array(separate_cells.make_movie().array.isel({AXIS.frames_dim: -1})),
        traces=PopSnap.from_array(separate_cells.traces.array.isel({AXIS.frames_dim: -1})),
    )

    expected = pix_stats.initialize(
        traces=separate_cells.traces, frames=separate_cells.make_movie()
    )
    assert expected == result


def test_ingest_component(pix_stats, separate_cells):
    pix_stats.initialize(
        traces=Traces.from_array(
            separate_cells.traces.array.isel({AXIS.component_dim: slice(None, -1)})
        ),
        frames=separate_cells.make_movie(),
    )

    result = pix_stats.ingest_component(
        frames=separate_cells.make_movie(),
        new_traces=Traces.from_array(separate_cells.traces.array.isel({AXIS.component_dim: [-1]})),
    )

    expected = pix_stats.initialize(
        traces=separate_cells.traces, frames=separate_cells.make_movie()
    )

    assert expected == result
