import pytest
import numpy as np
import xarray as xr
from cala.streaming.nodes.cold_start import ColdStarter, ColdStarterParams


class TestColdStart:
    @pytest.fixture(autouse=True, scope="class")
    def cold_starter(self):
        return ColdStarter(ColdStarterParams(num_frames=10, cell_radius=5))

    def test_estimate_gaussian_noise(self, cold_starter, stabilized_video):
        noise_level = cold_starter._estimate_gaussian_noise(
            stabilized_video, stabilized_video.shape
        )
        print(f"\nNoise Level: {noise_level}")

    @pytest.fixture(autouse=True, scope="class")
    def centered_video(self, cold_starter, stabilized_video):
        centered_video = cold_starter._center_to_median(stabilized_video)
        assert centered_video.max() < stabilized_video.max()
        return centered_video

    @pytest.fixture(autouse=True, scope="class")
    def new_components(self, cold_starter, centered_video, stabilized_video):
        energy_landscape = (centered_video**2).sum(dim=cold_starter.params.frames_axis)
        suspect_area = cold_starter._get_max_energy_slice(
            arr=stabilized_video, energy_landscape=energy_landscape
        )
        a_new, c_new = cold_starter._local_nmf(
            suspect_area,
            stabilized_video[0].shape,
            stabilized_video[0].dims,
            stabilized_video.coords[cold_starter.params.frames_axis].coords,
        )

        import matplotlib.pyplot as plt

        plt.imsave("mean_video.png", stabilized_video.mean(dim=cold_starter.params.frames_axis))
        plt.imsave("energy.png", energy_landscape)
        plt.imsave("a_new.png", a_new)
        plt.plot(c_new)
        plt.savefig("c_new.png")

        # same coordinates between a_new and np.max(energy_landscape) or np.max(stabilized_video.mean(dim=cold_starter.params.frames_axis))

        return a_new, c_new

    def test_validate_component(self, cold_starter, new_components, stabilized_video):
        a_new, c_new = new_components

        # a few different scenarios...
        # 1. completely new cell (cannot be rejected)
        # 2. same trace as existing - footprint overlapping (should merge post normalization)
        # 3. same trace as existing - footprint non-overlapping (should merge but how to normalize??)
        # 4. unique trace - footprint nonoverlapping ()
        # 5. unique trace - footprint overlapping ()

        # 1.
        footprints = xr.DataArray(
            np.array([]), dims=cold_starter.params.component_axis
        ).assign_coords()
        traces = xr.DataArray(np.array([]), dims=cold_starter.params.component_axis)

        valid = cold_starter._validate_new_components(
            new_footprints=a_new,
            new_traces=c_new,
            footprints=footprints,
            traces=traces,
            residuals=stabilized_video,
        )

        # 2.

        assert valid
