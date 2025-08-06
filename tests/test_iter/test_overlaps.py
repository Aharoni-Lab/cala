import numpy as np
import pytest
from noob.node import Node, NodeSpecification

from cala.assets import Footprints
from cala.models import AXIS


@pytest.fixture(scope="function")
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ov_init_test", type="cala.nodes.overlap.initialize")
    )


def test_init(init, separate_cells, connected_cells) -> None:
    overlap = init.process(footprints=separate_cells.footprints)

    assert np.trace(overlap.array) == len(separate_cells.cell_ids)
    assert np.all(np.triu(overlap.array, k=1) == 0)

    result = init.process(footprints=connected_cells.footprints)

    expected = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])

    np.testing.assert_array_equal(result.array, expected)


@pytest.fixture(scope="function")
def comp_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ov_init_test", type="cala.nodes.overlap.ingest_component")
    )


@pytest.mark.parametrize("toy", ["separate_cells", "connected_cells"])
def test_ingest_component(init, comp_update, toy, request) -> None:
    toy = request.getfixturevalue(toy)
    base = Footprints.from_array(toy.footprints.array.isel({AXIS.component_dim: slice(None, -1)}))
    new = Footprints.from_array(toy.footprints.array.isel({AXIS.component_dim: [-1]}))

    pre_ingest = init.process(footprints=base)

    result = comp_update.process(overlaps=pre_ingest, footprints=base, new_footprints=new)
    expected = init.process(footprints=toy.footprints)

    assert result == expected
