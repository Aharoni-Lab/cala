from cala.models import AXIS
from cala.nodes.prep.r_estimate import SizeEst
from cala.testing.toy import Position
from cala.util import package_frame


def test_size_estim(separate_cells):
    kwargs = {
        "min_sigma": 1,
        "max_sigma": 10,
        "num_sigma": 10,
        "threshold": 0.1,
        "overlap": 0.5,
    }
    node = SizeEst(n_frames=1, log_kwargs=kwargs)

    max_proj = package_frame(
        separate_cells.make_movie().array.max(dim=AXIS.frames_dim).values, index=1
    )
    result = node.get_median_radius(max_proj)

    expected = separate_cells.cell_radii[0] - 1
    assert result == expected
    assert len(node.sizes_) == 3

    max_proj = package_frame(
        separate_cells.make_movie().array.max(dim=AXIS.frames_dim).values, index=3
    )
    result = node.get_median_radius(max_proj)

    assert result == expected // 2 + 1
    assert len(node.sizes_) == 3

    for center in node.centers_:
        height = center[0].astype(int).item()
        width = center[1].astype(int).item()
        assert Position(width=width, height=height) in separate_cells.cell_positions
