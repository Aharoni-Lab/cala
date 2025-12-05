from cala.arrays import AXIS, Buffer
from cala.nodes.segment.cleanup import clear_overestimates


def test_clear_overestimates(single_cell) -> None:
    residual = Buffer.from_array(single_cell.make_movie().array, size=100)
    residual.array.loc[{AXIS.width_coord: slice(single_cell.cell_positions[0].width, None)}] *= -1

    result = clear_overestimates(
        footprints=single_cell.footprints, residuals=residual, nmf_error=-1.0
    )
    expected = single_cell.footprints.array.copy().as_numpy()
    expected.loc[{AXIS.width_coord: slice(single_cell.cell_positions[0].width, None)}] = 0

    assert result.equals(expected)
