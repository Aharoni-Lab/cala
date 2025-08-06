import numpy as np
import pytest
from noob.node import Node, NodeSpecification

from cala.assets import Frame, PopSnap, Traces
from cala.models import AXIS


@pytest.fixture
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="cs_init_test", type="cala.nodes.component_stats.initialize")
    )


def test_init(init, separate_cells) -> None:
    """Test the correctness of the component correlation computation."""
    result = init.process(separate_cells.traces)

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


@pytest.fixture
def frame_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="cs_frame_test", type="cala.nodes.component_stats.ingest_frame")
    )


def test_ingest_frame(init, frame_update, separate_cells) -> None:

    result = frame_update.process(
        component_stats=init.process(
            Traces.from_array(separate_cells.traces.array.isel({AXIS.frames_dim: slice(None, -1)}))
        ),
        frame=Frame.from_array(separate_cells.make_movie().array.isel({AXIS.frames_dim: -1})),
        new_traces=PopSnap.from_array(separate_cells.traces.array.isel({AXIS.frames_dim: -1})),
    )

    expected = init.process(separate_cells.traces)

    assert expected == result


@pytest.fixture
def comp_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(
            id="cs_comp_test", type="cala.nodes.component_stats.ingest_component"
        )
    )


def test_ingest_component(init, comp_update, separate_cells):

    result = comp_update.process(
        component_stats=init.process(
            Traces.from_array(
                separate_cells.traces.array.isel({AXIS.component_dim: slice(None, -1)})
            )
        ),
        traces=Traces.from_array(
            separate_cells.traces.array.isel({AXIS.component_dim: slice(None, -1)})
        ),
        new_trace=Traces.from_array(separate_cells.traces.array.isel({AXIS.component_dim: [-1]})),
    )

    expected = init.process(separate_cells.traces)

    assert expected == result
