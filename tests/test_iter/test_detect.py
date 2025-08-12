import numpy as np
import pytest
from noob.node import NodeSpecification

from cala.assets import Residual
from cala.nodes.detect import Cataloger, Sniffer, Energy, SliceNMF
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


@pytest.fixture(scope="class")
def energy_shape(energy, single_cell_video):
    return energy.process(Residual.from_array(single_cell_video.array), trigger=single_cell_video)


@pytest.fixture(scope="class")
def slice_nmf(toy):
    return SliceNMF.from_specification(
        spec=NodeSpecification(
            id="test_slice_nmf",
            type="cala.nodes.detect.SliceNMF",
            params={"cell_radius": 2 * toy.cell_radii[0], "validity_threshold": 0.8},
        )
    )


@pytest.fixture(scope="class")
def sniffer():
    return Sniffer.from_specification(
        spec=NodeSpecification(
            id="test_dupe_sniffer",
            type="cala.nodes.detect.Sniffer",
            params={"merge_threshold": 0.8},
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

    def test_check_validity(self, slice_nmf, single_cell_video, energy_shape, toy):
        """
        test this with multiple overlapping cells!!
        """
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video.array, energy_shape)
        footprint, _ = slice_nmf._local_nmf(
            slice_,
            toy.frame_dims.model_dump(),
        )
        assert slice_nmf._check_validity(footprint, single_cell_video.array)

    def test_process(self, slice_nmf, single_cell_video, energy_shape, toy):
        new_component = slice_nmf.process(
            Residual.from_array(single_cell_video.array), energy_shape
        )
        if new_component:
            new_fp, new_tr = new_component
        else:
            raise AssertionError("Failed to detect a new component")

        for new, old in zip([new_fp, new_tr], [toy.footprints, toy.traces]):
            assert_scalar_multiple_arrays(new.array, old.array)


class TestSniffer:
    @pytest.fixture(scope="class")
    def new_component(self, slice_nmf, single_cell_video, energy_shape):
        return slice_nmf.process(Residual.from_array(single_cell_video.array), energy_shape)

    def test_find_overlap_ids(self, toy, new_component, sniffer):
        new_fp, new_tr = new_component

        toy.add_cell(
            position=Position(width=260, height=260),
            radius=toy.cell_radii[0],
            trace=toy.traces.array[0][::-1].values,
            id_="cell_1",
        )

        ids = sniffer._find_overlap_ids(new_fp.array, toy.footprints.array)
        assert np.all(ids == ["cell_0", "cell_1"])

        toy.drop_cell("cell_1")

    def test_get_overlapped_traces(self, toy, new_component, sniffer):
        new_fp, _ = new_component
        ids = sniffer._find_overlap_ids(new_fp.array, toy.footprints.array)

        trace = sniffer._get_overlapped_traces(ids, toy.traces.array)

        assert np.all(trace == toy.traces.array)

    def test_has_unique_trace(self, toy, new_component, sniffer):
        new_fp, new_tr = new_component
        toy.add_cell(
            Position(width=260, height=260), 30, np.array(range(toy.n_frames, 0, -1)), "cell_1"
        )
        ids = sniffer._find_overlap_ids(new_fp.array, toy.footprints.array)
        traces = sniffer._get_overlapped_traces(ids, toy.traces.array)
        dupe = sniffer._get_synced_traces(new_tr.array, traces)

        assert len(dupe) == 1
        assert dupe[0][0] == "cell_0"
        assert np.isclose(dupe[0][1], 1.0)

        toy.drop_cell("cell_1")


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
