import numpy as np
import pytest
import xarray as xr
from noob.node import NodeSpecification
from sklearn.decomposition import NMF

from cala.assets import AXIS, Buffer
from cala.nodes.detect import SliceNMF
from cala.nodes.detect.slice_nmf import rank1nmf
from cala.testing.util import assert_scalar_multiple_arrays


@pytest.fixture(scope="module")
def slice_nmf():
    return SliceNMF.from_specification(
        spec=NodeSpecification(
            id="test_slice_nmf",
            type="cala.nodes.detect.SliceNMF",
            params={"min_frames": 10, "detect_thresh": 1, "reprod_tol": 0.001},
        )
    )


def test_process(slice_nmf, single_cell):
    new_component = slice_nmf.process(
        residuals=Buffer.from_array(single_cell.make_movie().array, size=100),
        energy=single_cell.make_movie().array.std(dim=AXIS.frames_dim),
        detect_radius=single_cell.cell_radii[0] * 2,
    )
    if new_component:
        new_fp, new_tr = new_component
    else:
        raise AssertionError("Failed to detect a new component")

    for new, old in zip([new_fp[0], new_tr[0]], [single_cell.footprints, single_cell.traces]):
        assert_scalar_multiple_arrays(new.array.as_numpy(), old.array.as_numpy())


def test_chunks(single_cell):
    nmf = SliceNMF.from_specification(
        spec=NodeSpecification(
            id="test_slice_nmf",
            type="cala.nodes.detect.SliceNMF",
            params={"min_frames": 10, "detect_thresh": 1, "reprod_tol": 0.001},
        )
    )
    fpts, trcs = nmf.process(
        residuals=Buffer.from_array(single_cell.make_movie().array, size=100),
        energy=single_cell.make_movie().array.std(dim=AXIS.frames_dim),
        detect_radius=10,
    )
    if not fpts or not trcs:
        raise AssertionError("Failed to detect a new component")

    factors = [trc.array.data.max() for trc in trcs]
    fpt_arr = xr.concat([f.array * m for f, m in zip(fpts, factors)], dim="component")

    expected = single_cell.footprints.array[0]
    result = fpt_arr.sum(dim="component")

    assert_scalar_multiple_arrays(expected.as_numpy(), result.as_numpy())
    for trc in trcs:
        assert_scalar_multiple_arrays(trc.array, single_cell.traces.array[0])


def test_rank1nmf(single_cell):
    Y = single_cell.make_movie().array
    R = Y.stack(space=AXIS.spatial_dims).transpose("space", AXIS.frames_dim)
    R += np.random.randint(0, 2, R.shape)

    shape = np.mean(R.values, axis=1).shape
    a_res, c_res, err_res = rank1nmf(R.values, np.random.random(shape), iters=10)

    nmf = NMF(n_components=1, init="random", max_iter=10, tol=1e-3)
    a_exp = nmf.fit_transform(R.values)
    c_exp = nmf.components_
    err_exp = nmf.reconstruction_err_

    assert_scalar_multiple_arrays(np.squeeze(a_exp), a_res)
    assert_scalar_multiple_arrays(np.squeeze(c_exp), c_res)
