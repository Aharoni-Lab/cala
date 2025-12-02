import numpy as np
import pytest
import xarray as xr
from noob.node import Node, NodeSpecification

from cala.assets import CompStats, Frame, PopSnap, Traces
from cala.models import AXIS


@pytest.fixture
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="cs_init_test", type="cala.nodes.omf.component_stats.initialize")
    )


def test_init(init, four_separate_cells) -> None:
    """Test the correctness of the component correlation computation."""
    result = init.process(four_separate_cells.traces.array)

    for id1, trace1 in zip(
        four_separate_cells.cell_ids,
        four_separate_cells.traces.array.transpose(AXIS.component_dim, ...),
    ):
        for id2, trace2 in zip(
            four_separate_cells.cell_ids,
            four_separate_cells.traces.array.transpose(AXIS.component_dim, ...),
        ):
            assert (
                result.set_xindex(AXIS.id_coord)
                .sel({AXIS.id_coord: id1})
                .set_xindex(f"{AXIS.id_coord}'")
                .sel({f"{AXIS.id_coord}'": id2})
                .item()
                == (trace1 @ trace2).item() / four_separate_cells.n_frames
            )

    # Test symmetry
    np.testing.assert_array_equal(result, result.T)


@pytest.fixture
def frame_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(
            id="cs_frame_test", type="cala.nodes.omf.component_stats.ingest_frame"
        )
    )


def test_ingest_frame(init, frame_update, four_separate_cells) -> None:

    result = frame_update.process(
        CompStats.from_array(
            init.process(four_separate_cells.traces.array.isel({AXIS.frames_dim: slice(None, -1)}))
        ),
        frame=Frame.from_array(four_separate_cells.make_movie().array.isel({AXIS.frames_dim: -1})),
        new_traces=PopSnap.from_array(four_separate_cells.traces.array.isel({AXIS.frames_dim: -1})),
    )

    expected = init.process(four_separate_cells.traces.array)

    xr.testing.assert_allclose(expected, result.array)


@pytest.fixture
def comp_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(
            id="cs_comp_test", type="cala.nodes.omf.component_stats.ingest_component"
        )
    )


def test_ingest_component(init, comp_update, four_separate_cells):
    slice_loc = -2

    result = comp_update.process(
        component_stats=CompStats.from_array(
            init.process(
                four_separate_cells.traces.array.isel({AXIS.component_dim: slice(None, slice_loc)})
            )
        ),
        traces=Traces.from_array(
            four_separate_cells.traces.array.isel({AXIS.component_dim: slice(None, slice_loc)})
        ),
        new_traces=Traces.from_array(
            four_separate_cells.traces.array.isel({AXIS.component_dim: slice(slice_loc, None)})
        ),
    )

    expected = init.process(four_separate_cells.traces.array)

    xr.testing.assert_allclose(expected, result.array)
