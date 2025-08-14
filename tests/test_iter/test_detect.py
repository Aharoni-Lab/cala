import numpy as np
import xarray as xr
import pytest
from noob.node import NodeSpecification

from cala.assets import Residual, AXIS
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
            params={"gaussian_std": 1.0, "min_frames": 10},
        )
    )


@pytest.fixture(scope="function")
def energy_shape(energy, single_cell_video):
    return energy.process(Residual.from_array(single_cell_video.array), trigger=single_cell_video)


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
        spec=NodeSpecification(id="test", type="cala.nodes.detect.Cataloger")
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
    @pytest.fixture(scope="class")
    def new_component(self, single_cell_video, energy_shape):
        return SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.slice_nmf.SliceNMF",
                params={"cell_radius": 60, "validity_threshold": 0.8},
            )
        ).process(Residual.from_array(single_cell_video.array), energy_shape)

    def test_register(self, cataloger, new_component, toy):
        new_fp, new_tr = new_component
        fp, tr = cataloger._register(
            new_fp=new_fp,
            new_tr=new_tr,
        )

        assert np.array_equal(fp.array, new_fp.array)
        assert np.array_equal(tr.array, new_tr.array)

    def test_merge(self, cataloger, toy, single_cell_video, energy_shape):
        new_component = SliceNMF.from_specification(
            spec=NodeSpecification(
                id="test_slice_nmf",
                type="cala.nodes.detect.slice_nmf.SliceNMF",
                params={"cell_radius": 10, "validity_threshold": 0.8},
            )
        ).process(Residual.from_array(single_cell_video.array), energy_shape)

        new_fp, new_tr = new_component
        fp, tr = cataloger._merge(
            new_fp, new_tr, toy.footprints, toy.traces, duplicates=[("cell_0", 1.0)]
        )

        movie_result = fp.array @ tr.array

        movie_new_comp = new_fp.array @ new_tr.array
        movie_expected = single_cell_video.array + movie_new_comp

        np.testing.assert_allclose(movie_result, movie_expected.transpose(*movie_result.dims))
