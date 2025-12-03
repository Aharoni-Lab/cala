import numpy as np
import pytest
from noob.node import Node, NodeSpecification

from cala.assets import AXIS
from cala.assets.assets import Footprint, Footprints, Overlaps


@pytest.fixture(scope="function")
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ov_init_test", type="cala.nodes.omf.overlap.initialize")
    )


def test_init(init, four_separate_cells, four_connected_cells) -> None:
    overlap = init.process(overlaps=Overlaps(), footprints=four_separate_cells.footprints)

    assert np.trace(overlap.array.as_numpy()) == len(four_separate_cells.cell_ids)
    assert np.all(np.triu(overlap.array.as_numpy(), k=1) == 0)

    result = init.process(overlaps=Overlaps(), footprints=four_connected_cells.footprints)

    expected = np.array([[1, 0, 1, 0], [0, 1, 1, 0], [1, 1, 1, 1], [0, 0, 1, 1]])

    np.testing.assert_array_equal(result.array.as_numpy(), expected)


@pytest.fixture(scope="function")
def comp_update() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="ov_init_test", type="cala.nodes.omf.overlap.ingest_component")
    )


@pytest.mark.parametrize("toy", ["four_separate_cells", "four_connected_cells"])
def test_ingest_component(init, comp_update, toy, request) -> None:
    """
    Need cases where a merge has occurred in Catalog (with 'replace' attr)

    """
    toy = request.getfixturevalue(toy)
    base = Footprints.from_array(toy.footprints.array.isel({AXIS.component_dim: slice(None, -1)}))
    new = Footprint.from_array(toy.footprints.array.isel({AXIS.component_dim: [-1]}))

    pre_ingest = init.process(overlaps=Overlaps(), footprints=base)

    result = comp_update.process(overlaps=pre_ingest, footprints=base, new_footprints=new)
    expected = init.process(overlaps=Overlaps(), footprints=toy.footprints)

    assert result == expected
