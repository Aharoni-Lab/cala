import pytest

from cala.assets import PopSnap, AXIS
from cala.gui.components.stamper import stamp


@pytest.mark.xfail
def test_deps_spec():
    """
    The spec file is correctly loaded, e.g. in the `/index` endpoint
    """
    raise NotImplementedError("Write me")


def test_stamper(connected_cells) -> None:
    fp = connected_cells.footprints
    trs = connected_cells.traces
    tr = PopSnap.from_array(trs.array.isel({AXIS.frames_dim: -1}))
    frame = stamp(fp, tr)

    assert len(frame.array.dims) == 3
