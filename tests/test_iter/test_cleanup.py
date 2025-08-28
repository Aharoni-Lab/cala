import xarray as xr

from cala.assets import Residual, Frame
from cala.models import AXIS
from cala.nodes.cleanup import clear_overestimates, _filter_redundant


def test_clear_overestimates(single_cell) -> None:
    residual = Residual.from_array(single_cell.make_movie().array)
    residual.array.loc[{AXIS.width_coord: slice(single_cell.cell_positions[0].width, None)}] *= -1

    result = clear_overestimates(
        footprints=single_cell.footprints, residuals=residual, nmf_error=-1.0
    )
    expected = single_cell.footprints.array.copy()
    expected.loc[{AXIS.width_coord: slice(single_cell.cell_positions[0].width, None)}] = 0

    assert result.equals(expected)


def test_erase_redundant(splitoff_cells) -> None:
    footprints = splitoff_cells.footprints
    dead_footprint = xr.DataArray(
        footprints.array.max(dim=AXIS.component_dim) / 100,
        dims=footprints.array.isel({AXIS.component_dim: 1}).dims,
        coords=footprints.array.isel({AXIS.component_dim: 1}).coords,
    ).assign_coords({AXIS.id_coord: "cell_2", AXIS.detect_coord: 77})
    footprints.array = xr.concat([footprints.array, dead_footprint], dim=AXIS.component_dim)

    traces = splitoff_cells.traces
    dead_trace = xr.DataArray(
        0.1,
        dims=traces.array.isel({AXIS.component_dim: 1}).dims,
        coords=traces.array.isel({AXIS.component_dim: 1}).coords,
    ).assign_coords({AXIS.id_coord: "cell_2", AXIS.detect_coord: 77})
    traces.array = xr.concat([traces.array, dead_trace], dim=AXIS.component_dim)

    frame = Frame.from_array(footprints.array @ traces.array.isel({AXIS.frames_dim: -1}))

    result = _filter_redundant(
        footprints=footprints, traces=traces, frame=frame, min_life_in_frames=10, quantile=0.9
    )

    expected = splitoff_cells.footprints.array[AXIS.id_coord].values.tolist()

    assert set(result) == set(expected)


def test_merge_components(splitoff_cells) -> None: ...
