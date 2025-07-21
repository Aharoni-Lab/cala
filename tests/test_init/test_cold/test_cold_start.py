import matplotlib.pyplot as plt
import numpy as np
import pytest

from cala.models.observable import Footprint, Footprints, Trace, Traces
from cala.nodes.init.cold.dupe_sniff import DupeSniffer, DupeSnifferParams
from cala.nodes.init.cold.energy import Energy, EnergyParams
from cala.nodes.init.cold.slice_nmf import SliceNMF, SliceNMFParams
from cala.nodes.init.cold.catalog import Cataloger, CatalogerParams
from cala.testing.toy import FrameSize, Position, Toy
from cala.testing.util import assert_scalar_multiple_arrays


@pytest.fixture(autouse=True, scope="module")
def simulator():
    n_frames = 300
    frame_dims = FrameSize(width=512, height=512)
    cell_positions = [Position(width=256, height=256)]
    cell_radii = 30
    cell_traces = [np.array(range(300), dtype=float)]
    confidences = [0.8]

    return Toy(
        n_frames=n_frames,
        frame_dims=frame_dims,
        cell_radii=cell_radii,
        cell_positions=cell_positions,
        cell_traces=cell_traces,
        confidences=confidences,
    )


@pytest.fixture(autouse=True, scope="module")
def single_cell_video(simulator):
    return simulator.make_movie()


class TestEnergy:
    @pytest.fixture(scope="class")
    def energy(self):
        return Energy(EnergyParams(gaussian_std=1.0))

    def test_estimate_gaussian_noise(self, energy, single_cell_video):
        noise_level = energy._estimate_gaussian_noise(single_cell_video, single_cell_video.shape)
        print(f"\nNoise Level: {noise_level}")

    def test_center_to_median(self, energy, single_cell_video):
        centered_video = energy._center_to_median(single_cell_video)
        assert centered_video.max() < single_cell_video.max()

    def test_process(self, energy, single_cell_video):
        energy_landscape = energy.process(single_cell_video)
        plt.imsave("mean_video.png", single_cell_video.mean(dim=energy.params.frames_dim))
        plt.imsave("energy.png", energy_landscape)
        assert energy_landscape.sizes == single_cell_video[0].sizes
        assert np.all(energy_landscape >= 0)


class TestSliceNMF:
    @pytest.fixture(scope="module")
    def energy_shape(self, single_cell_video):
        return Energy(EnergyParams(gaussian_std=1.0)).process(single_cell_video)

    @pytest.fixture(scope="class")
    def slice_nmf(self, simulator):
        return SliceNMF(
            SliceNMFParams(cell_radius=2 * simulator.cell_radii[0], validity_threshold=0.8)
        )

    def test_get_max_energy_slice(self, slice_nmf, single_cell_video, energy_shape):
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video, energy_shape)
        return slice_

    def test_local_nmf(self, slice_nmf, single_cell_video, energy_shape, simulator):
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video, energy_shape)
        footprint, trace = slice_nmf._local_nmf(
            slice_,
            simulator.frame_dims.model_dump(),
        )

        # Using the Pythagorean Theorem
        abab = (footprint @ simulator.footprints) ** 2
        aabb = footprint.dot(footprint) * simulator.footprints.dot(simulator.footprints)

        assert abab > aabb * 0.99999997

    def test_check_validity(self, slice_nmf, single_cell_video, energy_shape, simulator):
        """
        test this with multiple overlapping cells!!
        """
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video, energy_shape)
        footprint, _ = slice_nmf._local_nmf(
            slice_,
            simulator.frame_dims.model_dump(),
        )
        assert slice_nmf._check_validity(footprint, single_cell_video)

    def test_process(self, slice_nmf, single_cell_video, energy_shape, simulator):
        new_component = slice_nmf.process(single_cell_video, energy_shape)
        if new_component:
            new_fp, new_tr = new_component
        else:
            raise AssertionError("Failed to detect a new component")

        for new, old in zip([new_fp, new_tr], [simulator.footprints, simulator.traces]):
            assert_scalar_multiple_arrays(new, old)


class TestSniffer:
    @pytest.fixture(scope="module")
    def energy_shape(self, single_cell_video):
        return Energy(EnergyParams(gaussian_std=1.0)).process(single_cell_video)

    @pytest.fixture(scope="class")
    def new_component(self, single_cell_video, energy_shape):
        return SliceNMF(SliceNMFParams(cell_radius=10, validity_threshold=0.8)).process(
            single_cell_video, energy_shape
        )

    @pytest.fixture(scope="class")
    def sniffer(self):
        return DupeSniffer(params=DupeSnifferParams(merge_threshold=0.8))

    def test_find_overlap_ids(self, simulator, new_component, sniffer):
        new_fp, new_tr = new_component

        simulator.add_cell(
            Position(width=260, height=260),
            simulator.cell_radii[0],
            simulator.traces[0][::-1],
            "cell_1",
        )

        ids = sniffer._find_overlap_ids(new_fp, simulator.footprints)
        assert np.all(ids == ["cell_0", "cell_1"])

        simulator.drop_cell("cell_1")

    def test_get_overlapped_traces(self, simulator, new_component, sniffer):
        new_fp, _ = new_component
        ids = sniffer._find_overlap_ids(new_fp, simulator.footprints)

        trace = sniffer._get_overlapped_traces(ids, simulator.traces)

        assert np.all(trace == simulator.traces)

    def test_has_unique_trace(self, simulator, new_component, sniffer):
        new_fp, new_tr = new_component
        simulator.add_cell(
            Position(width=260, height=260), 30, np.array(range(300, 0, -1)), "cell_1"
        )
        ids = sniffer._find_overlap_ids(new_fp, simulator.footprints)
        traces = sniffer._get_overlapped_traces(ids, simulator.traces)
        dupe = sniffer._get_synced_traces(new_tr, traces)

        assert len(dupe) == 1
        assert dupe[0][0] == "cell_0"
        assert np.isclose(dupe[0][1], 1.0)

        simulator.drop_cell("cell_1")


class TestCataloger:
    @pytest.fixture(scope="module")
    def energy_shape(self, single_cell_video):
        return Energy(EnergyParams(gaussian_std=1.0)).process(single_cell_video)

    @pytest.fixture(scope="class")
    def new_component(self, single_cell_video, energy_shape):
        return SliceNMF(SliceNMFParams(cell_radius=60, validity_threshold=0.8)).process(
            single_cell_video, energy_shape
        )

    @pytest.fixture(scope="class")
    def cataloger(self):
        return Cataloger(params=CatalogerParams())

    def test_init_with(self, cataloger, new_component):
        fp, tr = cataloger._init_with(*new_component)

        assert np.array_equal(fp.values[0], new_component[0].values)
        assert np.array_equal(tr.values[0], new_component[1].values)

    def test_register(self, cataloger, new_component, simulator):
        new_fp, new_tr = new_component
        fp, tr = cataloger._register(
            new_fp=Footprint(array=new_fp),
            new_tr=Trace(array=new_tr),
            existing_fp=Footprints(array=simulator.footprints),
            existing_tr=Traces(array=simulator.traces),
        )

        assert np.array_equal(fp.array.values[-1], new_component[0].values)
        assert np.array_equal(tr.array.values[-1], new_component[1].values)

    def test_merge(self, cataloger, simulator, single_cell_video, energy_shape):
        new_component = SliceNMF(SliceNMFParams(cell_radius=10, validity_threshold=0.8)).process(
            single_cell_video, energy_shape
        )
        new_fp, new_tr = new_component
        fp, tr = cataloger._merge(
            Footprint(array=new_fp),
            Trace(array=new_tr),
            Footprints(array=simulator.footprints),
            Traces(array=simulator.traces),
            duplicates=[("cell_0", 1.0)],
        )

        movie_recon = fp.array @ tr.array
        movie_new_comp = new_fp @ new_tr
        movie_expected = single_cell_video + movie_new_comp

        np.testing.assert_allclose(movie_recon, movie_expected.transpose(*movie_recon.dims))
