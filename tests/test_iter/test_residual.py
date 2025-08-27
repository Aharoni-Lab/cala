import numpy as np
import pytest
import xarray as xr
from noob.node import Node, NodeSpecification

from cala.assets import Residual
from cala.models.axis import AXIS
from cala.nodes.residual import _align_overestimates, _find_unlayered_footprints


@pytest.fixture(scope="function")
def init() -> Node:
    return Node.from_specification(
        spec=NodeSpecification(id="res_init_test", type="cala.nodes.residual.build")
    )


def test_init(init, separate_cells) -> None:
    result = init.process(
        residuals=Residual(),
        footprints=separate_cells.footprints,
        traces=separate_cells.traces,
        frames=separate_cells.make_movie(),
    )

    assert np.all(result.array == 0)


def test_align_overestimates(single_cell) -> None:
    """
    grab the last frame of the residual. assume part of the footprint masked area is negative
    traces needs to proportionally decrease, until the recalculated residual is zero

    Eventually, this probably can be absorbed straight into trace frame_ingest as a constraint.
    """
    movie = single_cell.make_movie()
    last_frame = movie.array.isel({AXIS.frames_dim: -1})

    last_res = xr.zeros_like(last_frame)
    last_res.loc[{AXIS.width_coord: slice(single_cell.cell_positions[0].width, None)}] = -1
    last_res = last_res.where(single_cell.footprints.array[0].values, 0)

    last_trace = single_cell.traces.array.isel({AXIS.frames_dim: -1})

    footprints = single_cell.footprints.array

    adjusted_traces = _align_overestimates(A=footprints, R_latest=last_res, C_latest=last_trace)

    result = (footprints @ adjusted_traces).values
    expected = movie.array.isel({AXIS.frames_dim: -2}).values

    np.testing.assert_array_equal(result, expected)


def test_find_exposed_footprints(connected_cells) -> None:
    footprints = connected_cells.footprints
    result = _find_unlayered_footprints(footprints.array)
    assert result.sum(dim=AXIS.component_dim).max().item() == footprints.array.max().item()
