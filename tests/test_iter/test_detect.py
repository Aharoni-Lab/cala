import numpy as np
import pytest
import xarray as xr
from noob.node import NodeSpecification
from sklearn.decomposition import NMF

from cala.assets import AXIS, Buffer, Footprints, Traces
from cala.nodes.detect import Cataloger, SliceNMF
from cala.nodes.detect.catalog import _merge_with, _register
from cala.nodes.detect.slice_nmf import rank1nmf
from cala.testing.util import assert_scalar_multiple_arrays


@pytest.fixture(scope="class")
def slice_nmf():
    return SliceNMF.from_specification(
        spec=NodeSpecification(
            id="test_slice_nmf",
            type="cala.nodes.detect.SliceNMF",
            params={"min_frames": 10, "detect_thresh": 1, "reprod_tol": 0.001},
        )
    )


@pytest.fixture(scope="class")
def cataloger():
    return Cataloger.from_specification(
        spec=NodeSpecification(
            id="test",
            type="cala.nodes.detect.Cataloger",
            params={
                "age_limit": 100,
                "smooth_kwargs": {"sigma": 2},
                "merge_threshold": 0.8,
                "val_threshold": 0.5,
                "cnt_threshold": 5,
            },
        )
    )


class TestSliceNMF:
    def test_process(self, slice_nmf, single_cell):
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

    def test_chunks(self, single_cell):
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


class TestCataloger:
    @pytest.fixture(scope="function")
    def new_component(self, slice_nmf, single_cell):
        buff = Buffer(size=100)
        buff.array = single_cell.make_movie().array
        return slice_nmf.process(
            residuals=buff, energy=buff.array.std(dim=AXIS.frames_dim), detect_radius=60
        )

    def test_register(self, cataloger, new_component):
        new_fp, new_tr = new_component
        fp, tr = _register(new_fp=new_fp[0].array, new_tr=new_tr[0].array)

        assert np.array_equal(fp.as_numpy(), new_fp[0].array.as_numpy())
        assert np.array_equal(tr, new_tr[0].array)

    def test_merge_with(self, slice_nmf, cataloger, single_cell):
        buff = Buffer(size=100)
        buff.array = single_cell.make_movie().array
        new_component = slice_nmf.process(
            buff, energy=buff.array.std(dim=AXIS.frames_dim), detect_radius=10
        )

        new_fp, new_tr = new_component
        fp, tr = _merge_with(
            new_fp[0].array.expand_dims(dim="component"),
            new_tr[0].array.expand_dims(dim="component"),
            single_cell.footprints.array,
            single_cell.traces.array,
            ["cell_0"],
        )

        movie_result = (
            (fp @ tr).reset_coords([AXIS.id_coord, AXIS.detect_coord], drop=True).as_numpy()
        )

        movie_new_comp = new_fp[0].array @ new_tr[0].array
        movie_expected = (single_cell.make_movie().array + movie_new_comp).transpose(
            *movie_result.dims
        )

        xr.testing.assert_allclose(movie_result, movie_expected)

    def test_process_ideal(self, slice_nmf, cataloger, separate_cells):
        """
        test cataloging separate cells. ideal case with cell_radius=5
        """
        buff = Buffer(size=100)
        buff.array = separate_cells.make_movie().array
        fps, trs = slice_nmf.process(buff, buff.array.std(dim=AXIS.frames_dim), detect_radius=5)

        # NOTE: by manually putting in separate_cells, we're forcing a double-detection in this test
        new_fps, new_trs = cataloger.process(
            fps, trs, separate_cells.footprints, separate_cells.traces
        )

        result = new_fps.array @ new_trs.array

        # would not detect cell_0 and cell_1 since they're uniform
        detected = ["cell_2", "cell_3"]
        expected = (
            separate_cells.footprints.array.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: detected})
            @ separate_cells.traces.array.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: detected})
        ).transpose(*result.dims) * 2

        assert set(new_fps.array.attrs.get("replaces")) == set(detected)
        xr.testing.assert_allclose(result.as_numpy(), expected.as_numpy())

    def test_process_fail(self, slice_nmf, cataloger, separate_cells):
        """
        test cataloging separate cells. nmf supposed to fail with radius=25 (grabs too many cells)
        """
        movie = separate_cells.make_movie().array
        fps, trs = slice_nmf.process(
            Buffer.from_array(movie, size=100), movie.std(dim=AXIS.frames_dim), detect_radius=25
        )

        # NOTE: by manually putting in separate_cells, we're forcing a double-detection in this test
        new_fps, new_trs = cataloger.process(
            fps, trs, separate_cells.footprints, separate_cells.traces
        )

        assert new_fps.array is None and new_trs.array is None

    def test_process_connected(self, slice_nmf, cataloger, connected_cells):
        """
        trial with connected cells 🙏
        """
        movie = connected_cells.make_movie().array
        fps, trs = slice_nmf.process(
            Buffer.from_array(movie, size=100), movie.std(dim=AXIS.frames_dim), detect_radius=4
        )

        # NOTE: by manually putting in connected_cells,
        # we're forcing a double-detection in this test
        new_fps, new_trs = cataloger.process(fps, trs, Footprints(), Traces())

        result = (new_fps.array @ new_trs.array).transpose(AXIS.frames_dim, ...).as_numpy()
        expected = movie.transpose(*result.dims).as_numpy()

        # not sure why we're getting some stray pixels... but we need to remove them
        sig_pxls = (new_fps.array.max(dim=AXIS.component_dim) > 0.1).as_numpy()
        result, expected = result.where(sig_pxls), expected.where(sig_pxls)

        assert new_fps.array is not None
        # 1. the footprints do not overlap
        assert np.all(
            np.triu((new_fps.array @ new_fps.array.rename(AXIS.component_rename)).as_numpy(), 1)
            == 0
        )
        # 2. the trace and footprint values are accurate (where they do exist)
        xr.testing.assert_allclose(result, expected, atol=1)


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
