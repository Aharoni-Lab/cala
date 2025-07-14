import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

from cala.streaming.nodes.init.cold import Energy, EnergyParams, SliceNMF, SliceNMFParams
from cala.testing.simulation import Simulator, FrameSize, Position


@pytest.fixture(autouse=True, scope="module")
def simulator():
    n_frames = 300
    frame_dims = FrameSize(width=512, height=512)
    cell_positions = [Position(width=256, height=256)]
    cell_radii = 30
    cell_traces = [np.array(range(300))]

    return Simulator(
        n_frames=n_frames,
        frame_dims=frame_dims,
        cell_radii=cell_radii,
        cell_positions=cell_positions,
        cell_traces=cell_traces,
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
        plt.imsave("mean_video.png", single_cell_video.mean(dim=energy.params.frames_axis))
        plt.imsave("energy.png", energy_landscape)
        assert energy_landscape.sizes == single_cell_video[0].sizes
        assert np.all(energy_landscape >= 0)


class TestSliceNMF:
    @pytest.fixture(scope="module")
    def energy_shape(self, single_cell_video):
        return Energy(EnergyParams(gaussian_std=1.0)).process(single_cell_video)

    @pytest.fixture(scope="class")
    def slice_nmf(self):
        return SliceNMF(SliceNMFParams(cell_radius=10, validity_threshold=0.8))

    def test_get_max_energy_slice(self, slice_nmf, single_cell_video, energy):
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video, energy)
        return slice_

    def test_local_nmf(self, slice_nmf, single_cell_video, energy_shape, simulator):
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video, energy_shape)
        footprint, trace = slice_nmf._local_nmf(
            slice_,
            simulator.frame_dims.model_dump(),
        )
        print(f"footprint: {footprint.sizes}")
        print(f"trace: {trace.sizes}")
        plt.imsave("a_new.png", footprint)
        plt.plot(trace)
        plt.savefig("c_new.png")

    def test_check_validity(self, slice_nmf, single_cell_video, energy_shape, simulator):
        slice_ = slice_nmf._get_max_energy_slice(single_cell_video, energy_shape)
        footprint, _ = slice_nmf._local_nmf(
            slice_,
            simulator.frame_dims.model_dump(),
        )
        assert slice_nmf._check_validity(footprint, single_cell_video)

    def test_process(self, slice_nmf, single_cell_video, energy_shape):
        new_component = slice_nmf.process(single_cell_video, energy_shape)
        if new_component:
            new_fp, new_tr = new_component


class TestCataloger:
    @pytest.fixture(scope="class")
    def new_component(self, single_cell_video, energy_shape):
        return SliceNMF(SliceNMFParams(cell_radius=10, validity_threshold=0.8)).process(
            single_cell_video, energy_shape
        )


class TestColdStart:

    def test_validate_component(self, cold_starter, new_components, single_cell_video):
        a_new, c_new = new_components

        # a few different scenarios...
        # 1. very first cell (cannot be rejected)
        # 2. same trace as existing - footprint overlapping (should merge)
        # 3. same trace as existing - footprint non-overlapping (add as a new cell - could be just correlated neurons)
        # 4. unique trace - footprint nonoverlapping (add as a new component)
        # 5. unique trace - footprint overlapping (add as a new component)

        # 1. completely new cell (cannot be rejected)
        starting_footprints = xr.DataArray(
            np.array([]), dims=cold_starter.params.component_axis
        ).assign_coords()
        starting_traces = xr.DataArray(np.array([]), dims=cold_starter.params.component_axis)

        valid = cold_starter._validate_new_component(
            new_footprint=a_new,
            new_trace=c_new,
            footprints=starting_footprints,
            traces=starting_traces,
            residuals=single_cell_video,
        )

        assert valid

    @pytest.fixture
    def starter_with_one_component(self, cold_starter, new_components, single_cell_video):
        a_new, c_new = new_components
        new_component = a_new * c_new

        cold_starter.residuals_ = single_cell_video - new_component

        cold_starter.new_footprints_.append(a_new)
        cold_starter.new_traces_.append(c_new)

        return cold_starter

    def test_validate_duplicate_component(self, starter_with_one_component):
        # 2. same trace as existing - footprint overlapping (should merge)

        # Compute deviation from median
        V = starter_with_one_component._center_to_median(starter_with_one_component.residuals_)

        # Compute energy (variance)
        E = (V**2).sum(dim="frame")

        # Find and analyze neighborhood of maximum variance
        neighborhood = starter_with_one_component._get_max_energy_slice(
            arr=starter_with_one_component.residuals_, energy_landscape=E
        )
        a_merge, c_merge = starter_with_one_component._local_nmf(
            slice_=neighborhood,
            spatial_sizes=starter_with_one_component.residuals_.sel({"frame": 0}).sizes,
            temporal_coords=starter_with_one_component.residuals_.coords,
        )

        plt.imsave("a_merge.png", a_merge)

        valid = starter_with_one_component._validate_new_component(
            new_footprint=a_merge,
            new_trace=c_merge,
            footprints=starter_with_one_component.new_footprints_,
            traces=starter_with_one_component.new_traces_,
            residuals=starter_with_one_component.residuals_,
        )

        assert valid
