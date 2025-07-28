import numpy as np
import pytest
from noob.node import NodeSpecification

from cala.models import AXIS, Frame, PopSnap, Traces
from cala.nodes.iter.component_stats import CompStater
from cala.testing.toy import FrameDims, Position, Toy


@pytest.fixture(scope="function")
def comp_stats() -> CompStater:
    return CompStater.from_specification(
        spec=NodeSpecification(
            id="comp-stat-test", type="cala.nodes.iter.component_stats.CompStater"
        )
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


def test_init(comp_stats, separate_cells) -> None:
    """Test the correctness of the component correlation computation."""
    result = comp_stats.initialize(separate_cells.traces)

    for id1, trace1 in zip(
        separate_cells.cell_ids, separate_cells.traces.array.transpose(AXIS.component_dim, ...)
    ):
        for id2, trace2 in zip(
            separate_cells.cell_ids, separate_cells.traces.array.transpose(AXIS.component_dim, ...)
        ):
            assert (
                result.array.set_xindex(AXIS.id_coord)
                .sel({AXIS.id_coord: id1})
                .set_xindex(f"{AXIS.id_coord}'")
                .sel({f"{AXIS.id_coord}'": id2})
                .item()
                == (trace1 @ trace2).item() / separate_cells.n_frames
            )

    # Test symmetry
    np.testing.assert_array_equal(result.array, result.array.T)


def test_ingest_frame(comp_stats, separate_cells) -> None:

    comp_stats.initialize(
        Traces(array=separate_cells.traces.array.isel({AXIS.frames_dim: slice(None, -1)}))
    )

    result = comp_stats.ingest_frame(
        frame=Frame(array=separate_cells.make_movie().array.isel({AXIS.frames_dim: -1})),
        new_traces=PopSnap(array=separate_cells.traces.array.isel({AXIS.frames_dim: -1})),
    )

    expected = comp_stats.initialize(separate_cells.traces)

    assert expected == result


def test_ingest_component(comp_stats, separate_cells):
    comp_stats.initialize(
        Traces(array=separate_cells.traces.array.isel({AXIS.component_dim: slice(None, -1)}))
    )

    result = comp_stats.ingest_component(
        traces=Traces(
            array=separate_cells.traces.array.isel({AXIS.component_dim: slice(None, -1)})
        ),
        new_traces=Traces(array=separate_cells.traces.array.isel({AXIS.component_dim: [-1]})),
    )

    expected = comp_stats.initialize(separate_cells.traces)

    assert expected == result
