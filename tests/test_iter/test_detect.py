import numpy as np
import xarray as xr
import pytest
from noob.node import NodeSpecification

from cala.assets import Residual, AXIS, Footprints, Traces
from cala.nodes.detect import Cataloger, Energy, SliceNMF
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


@pytest.fixture(autouse=True, scope="class")
def single_cell_video(toy):
    return toy.make_movie()


@pytest.fixture(scope="class")
def energy():
    return Energy.from_specification(
        spec=NodeSpecification(
            id="test_energy",
            type="cala.nodes.detect.Energy",
            params={"min_frames": 10},
        )
    )


@pytest.fixture(scope="function")
def energy_shape(energy, single_cell_video):
    return energy.process(Residual.from_array(single_cell_video.array), trigger=True)


@pytest.fixture(scope="class")
def slice_nmf(toy):
    return SliceNMF.from_specification(
        spec=NodeSpecification(
            id="test_slice_nmf",
            type="cala.nodes.detect.SliceNMF",
            params={"cell_radius": 2 * toy.cell_radii[0]},
        )
    )


@pytest.fixture(scope="class")
def cataloger():
    return Cataloger.from_specification(
        spec=NodeSpecification(
            id="test", type="cala.nodes.detect.Cataloger", params={"merge_threshold": 0.8}
        )
    )


class TestEnergy:
    def test_estimate_gaussian_noise(self, energy, single_cell_video):
        noise_level = energy._estimate_gaussian_noise(single_cell_video.array)
        print(f"\nNoise Level: {noise_level}")

    def test_center_to_median(self, energy, single_cell_video):
        centered_video = energy._center_to_median(single_cell_video.array)
        assert centered_video.max() < single_cell_video.array.max()

    def test_process(self, energy, single_cell_video):
        energy_landscape = energy.process(
            residuals=Residual.from_array(single_cell_video.array), trigger=single_cell_video
        )
        assert energy_landscape.sizes == single_cell_video.array[0].sizes
        assert np.all(energy_landscape >= 0)


class TestSliceNMF:
    def test_get_max_energy_slice(self, slice_nmf, single_cell_video, energy_shape):
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video.array, energy_shape)
        return slice_

    def test_local_nmf(self, slice_nmf, single_cell_video, energy_shape, toy):
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video.array, energy_shape)
        footprint, trace = slice_nmf._local_nmf(
            slice_,
            toy.frame_dims.model_dump(),
        )

        assert_scalar_multiple_arrays(footprint, toy.footprints.array)

    def test_process(self, slice_nmf, single_cell_video, energy_shape, toy):
        new_component = slice_nmf.process(
            Residual.from_array(single_cell_video.array), energy_shape
        )
        if new_component:
            new_fp, new_tr = new_component
        else:
            raise AssertionError("Failed to detect a new component")

        for new, old in zip([new_fp[0], new_tr[0]], [toy.footprints, toy.traces]):
            assert_scalar_multiple_arrays(new.array, old.array)

    def test_chunks(self, single_cell_video, energy_shape, toy):
        nmf = SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.SliceNMF",
                params={"cell_radius": 10},
            )
        )
        fpts, trcs = nmf.process(Residual.from_array(single_cell_video.array), energy_shape)
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
    def new_component(self, single_cell_video, energy_shape):
        return SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.slice_nmf.SliceNMF",
                params={"cell_radius": 60},
            )
        ).process(Residual.from_array(single_cell_video.array), energy_shape)

    def test_register(self, cataloger, new_component, toy):
        new_fp, new_tr = new_component
        fp, tr = cataloger._register(
            new_fp=new_fp[0],
            new_tr=new_tr[0],
        )

        assert np.array_equal(fp.array, new_fp[0].array)
        assert np.array_equal(tr.array, new_tr[0].array)

    def test_merge_with(self, cataloger, toy, single_cell_video, energy_shape):
        new_component = SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.slice_nmf.SliceNMF",
                params={"cell_radius": 10},
            )
        ).process(Residual.from_array(single_cell_video.array), energy_shape)

        new_fp, new_tr = new_component
        fp, tr = cataloger._merge_with(
            new_fp[0].array, new_tr[0].array, toy.footprints.array, toy.traces.array, ["cell_0"]
        )

        movie_result = (fp.array @ tr.array).reset_coords(
            [AXIS.id_coord, AXIS.confidence_coord], drop=True
        )

        movie_new_comp = new_fp[0].array @ new_tr[0].array
        movie_expected = (single_cell_video.array + movie_new_comp).transpose(*movie_result.dims)

        xr.testing.assert_allclose(movie_result, movie_expected)

    def test_process_ideal(self, cataloger, separate_cells, energy):
        """
        test cataloging separate cells. ideal case with cell_radius=5
        """
        movie = separate_cells.make_movie().array
        ener = energy.process(Residual.from_array(movie), trigger=True)
        fps, trs = SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.slice_nmf.SliceNMF",
                params={"cell_radius": 5},
            )
        ).process(Residual.from_array(movie), ener)

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

    def test_process_fail(self, cataloger, separate_cells, energy):
        """
        test cataloging separate cells. nmf supposed to fail with radius=25 (grabs too many cells)
        """
        movie = separate_cells.make_movie().array
        ener = energy.process(Residual.from_array(movie), trigger=True)
        fps, trs = SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.slice_nmf.SliceNMF",
                params={"cell_radius": 25},
            )
        ).process(Residual.from_array(movie), ener)

        # NOTE: by manually putting in separate_cells, we're forcing a double-detection in this test
        new_fps, new_trs = cataloger.process(
            fps, trs, separate_cells.footprints, separate_cells.traces
        )

        assert new_fps.array is None and new_trs.array is None

    def test_process_connected(self, cataloger, connected_cells, energy):
        """
        trial with connected cells 🙏
        """
        movie = connected_cells.make_movie().array
        ener = energy.process(Residual.from_array(movie), trigger=True)
        fps, trs = SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.slice_nmf.SliceNMF",
                params={"cell_radius": 4},
            )
        ).process(Residual.from_array(movie), ener)

        # NOTE: by manually putting in connected_cells, we're forcing a double-detection in this test
        new_fps, new_trs = cataloger.process(fps, trs, Footprints(), Traces())

        result = (new_fps.array @ new_trs.array).where(new_fps.array.max(dim=AXIS.component_dim), 0)
        expected = movie.where(new_fps.array.max(dim=AXIS.component_dim), 0)

        assert new_fps.array is not None
        # 1. the footprints do not overlap
        assert np.all(np.triu(new_fps.array @ new_fps.array.rename(AXIS.component_rename), 1) == 0)
        # 2. the trace and footprint values are accurate (where they do exist)
        xr.testing.assert_allclose(result, expected.transpose(*result.dims), atol=1e-5)
