import numpy as np
import pytest
import xarray as xr
from noob.node import NodeSpecification

from cala.assets import AXIS, Footprints, Residual, Traces
from cala.nodes.detect import Cataloger, SliceNMF
from cala.testing.toy import FrameDims, Position, Toy
from cala.testing.util import assert_scalar_multiple_arrays


@pytest.fixture(autouse=True, scope="module")
def toy():
    n_frames = 30
    frame_dims = FrameDims(width=512, height=512)
    cell_positions = [Position(width=256, height=256)]
    cell_radii = 30
    cell_traces = [np.array(range(n_frames), dtype=float)]
    confidences = [0.8]

    return Toy(
        n_frames=n_frames,
        frame_dims=frame_dims,
        cell_radii=cell_radii,
        cell_positions=cell_positions,
        cell_traces=cell_traces,
        confidences=confidences,
    )


@pytest.fixture(scope="class")
def slice_nmf():
    return SliceNMF.from_specification(
        spec=NodeSpecification(
            id="test_slice_nmf",
            type="cala.nodes.detect.SliceNMF",
            params={"min_frames": 10, "detect_thresh": 1},
        )
    )


@pytest.fixture(scope="class")
def cataloger():
    return Cataloger.from_specification(
        spec=NodeSpecification(
            id="test", type="cala.nodes.detect.Cataloger", params={"merge_threshold": 0.8}
        )
    )


class TestSliceNMF:
    def test_process(self, slice_nmf, toy):
        new_component = slice_nmf.process(
            Residual.from_array(toy.make_movie().array),
            detect_radius=toy.cell_radii[0] * 2,
        )
        if new_component:
            new_fp, new_tr = new_component
        else:
            raise AssertionError("Failed to detect a new component")

        for new, old in zip([new_fp[0], new_tr[0]], [toy.footprints, toy.traces]):
            assert_scalar_multiple_arrays(new.array, old.array)

    def test_chunks(self, toy):
        nmf = SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.SliceNMF",
                params={"min_frames": 10, "detect_thresh": 1},
            )
        )
        fpts, trcs = nmf.process(Residual.from_array(toy.make_movie().array), detect_radius=10)
        if not fpts or not trcs:
            raise AssertionError("Failed to detect a new component")

        fpt_arr = xr.concat([f.array for f in fpts], dim="component")

        expected = toy.footprints.array[0]
        result = (fpt_arr.sum(dim="component") > 0).astype(int)

        assert np.array_equal(expected, result)
        for trc in trcs:
            assert_scalar_multiple_arrays(trc.array, toy.traces.array[0])


class TestCataloger:
    @pytest.fixture(scope="function")
    def new_component(self, slice_nmf, toy):
        return slice_nmf.process(Residual.from_array(toy.make_movie().array), detect_radius=60)

    def test_register(self, cataloger, new_component, toy):
        new_fp, new_tr = new_component
        fp, tr = cataloger._register(
            new_fp=new_fp[0].array,
            new_tr=new_tr[0].array,
        )

        assert np.array_equal(fp.array, new_fp[0].array)
        assert np.array_equal(tr.array, new_tr[0].array)

    def test_merge_with(self, slice_nmf, cataloger, toy):
        new_component = slice_nmf.process(
            Residual.from_array(toy.make_movie().array), detect_radius=10
        )

        new_fp, new_tr = new_component
        fp, tr = cataloger._merge_with(
            new_fp[0].array, new_tr[0].array, toy.footprints.array, toy.traces.array, ["cell_0"]
        )

        movie_result = (fp.array @ tr.array).reset_coords(
            [AXIS.id_coord, AXIS.confidence_coord], drop=True
        )

        movie_new_comp = new_fp[0].array @ new_tr[0].array
        movie_expected = (toy.make_movie().array + movie_new_comp).transpose(*movie_result.dims)

        xr.testing.assert_allclose(movie_result, movie_expected)

    def test_process_ideal(self, slice_nmf, cataloger, separate_cells):
        """
        test cataloging separate cells. ideal case with cell_radius=5
        """
        movie = separate_cells.make_movie().array
        fps, trs = slice_nmf.process(Residual.from_array(movie), detect_radius=5)

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

        assert new_fps.array.attrs.get("replaces") == detected
        xr.testing.assert_allclose(result, expected)

    def test_process_fail(self, slice_nmf, cataloger, separate_cells):
        """
        test cataloging separate cells. nmf supposed to fail with radius=25 (grabs too many cells)
        """
        movie = separate_cells.make_movie().array
        fps, trs = slice_nmf.process(Residual.from_array(movie), detect_radius=25)

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
        fps, trs = slice_nmf.process(Residual.from_array(movie), detect_radius=4)

        # NOTE: by manually putting in connected_cells,
        # we're forcing a double-detection in this test
        new_fps, new_trs = cataloger.process(fps, trs, Footprints(), Traces())

        result = (new_fps.array @ new_trs.array).where(new_fps.array.max(dim=AXIS.component_dim), 0)
        expected = movie.where(new_fps.array.max(dim=AXIS.component_dim), 0)

        assert new_fps.array is not None
        # 1. the footprints do not overlap
        assert np.all(np.triu(new_fps.array @ new_fps.array.rename(AXIS.component_rename), 1) == 0)
        # 2. the trace and footprint values are accurate (where they do exist)
        xr.testing.assert_allclose(result, expected.transpose(*result.dims), atol=1e-5)
