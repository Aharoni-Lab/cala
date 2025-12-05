import pytest

from cala.arrays import AXIS, PopSnap
from cala.gui.components.stamper import stamp


@pytest.mark.xfail
def test_deps_spec():
    """
    The spec file is correctly loaded, e.g. in the `/index` endpoint
    """
    raise NotImplementedError("Write me")


def test_stamper(four_connected_cells) -> None:
    fp = four_connected_cells.footprints
    trs = four_connected_cells.traces
    tr = PopSnap.from_array(trs.array.isel({AXIS.frame_dim: -1}))
    frame = stamp(fp, tr, gain=1)

    assert len(frame.array.dims) == 3
