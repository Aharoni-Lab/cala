import numpy as np
import pytest
from noob.node import NodeSpecification

from cala.models import AXIS, Frame, Movie, PopSnap, Traces
from cala.nodes.iter.residual import Resident
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture(scope="function")
def resident() -> Resident:
    return Resident.from_specification(
        spec=NodeSpecification(id="resident_test", type="cala.nodes.iter.residual.Resident")
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


def test_init(resident, separate_cells) -> None:
    result = resident.initialize(
        footprints=separate_cells.footprints,
        traces=separate_cells.traces,
        frames=separate_cells.make_movie(),
    )

    assert np.all(result.array == 0)


def test_ingest_frame(resident, separate_cells) -> None:

    resident.initialize(
        footprints=separate_cells.footprints,
        traces=Traces(array=separate_cells.traces.array.isel({AXIS.frames_dim: slice(None, -1)})),
        frames=Movie(
            array=separate_cells.make_movie().array.isel({AXIS.frames_dim: slice(None, -1)})
        ),
    )

    residual = resident.ingest_frame(
        frame=Frame(array=separate_cells.make_movie().array.isel({AXIS.frames_dim: -1})),
        footprints=separate_cells.footprints,
        traces=PopSnap(array=separate_cells.traces.array.isel({AXIS.frames_dim: -1})),
    )

    assert np.all(residual.array == 0)
