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
            stabilized_video[0].sizes,
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

    def test_validate_component(
        self, cold_starter, new_components, stabilized_video, footprints, traces
    ):
        a_new, c_new = new_components

        # a few different scenarios...
        # 1. very first cell (cannot be rejected)
        # 2. same trace as existing - footprint overlapping (should merge post normalization)
        # 3. same trace as existing - footprint non-overlapping (should merge but how to normalize??)
        # 4. unique trace - footprint nonoverlapping (add as a new component)
        # 5. unique trace - footprint overlapping (add as a new component)

        # 1. completely new cell (cannot be rejected)
        starting_footprints = xr.DataArray(
            np.array([]), dims=cold_starter.params.component_axis
        ).assign_coords()
        starting_traces = xr.DataArray(np.array([]), dims=cold_starter.params.component_axis)

        valid = cold_starter._validate_new_components(
            new_footprints=a_new,
            new_traces=c_new,
            footprints=starting_footprints,
            traces=starting_traces,
            residuals=stabilized_video,
        )

        assert valid

        new_component = a_new * c_new
        cold_starter.residuals_ = stabilized_video - new_component

        cold_starter.new_footprints_.append(a_new)
        cold_starter.new_traces_.append(c_new)

        # 2. same trace as existing - footprint overlapping (should merge post normalization)
        def idx_from_trace(orig_traces: xr.DataArray, new_trace: xr.DataArray) -> int:
            return xr.corr(new_trace, orig_traces, dim="frame").argmax()

        import matplotlib.pyplot as plt

        idx = idx_from_trace(traces, c_new)
        overlap_fp = footprints.sel({"component": idx})
        plt.imsave("overlap.png", overlap_fp)

        # fp_cutoff = overlap_fp.where(overlap_fp > 0, drop=True).quantile(0.90)
        centroid = overlap_fp.reset_coords(["id_", "type_"], drop=True).where(a_new > 0, 0)
        rim = overlap_fp.reset_coords(["id_", "type_"], drop=True).where(a_new <= 0, 0).copy()
        footprints.loc[{"component": idx}] = centroid

        plt.imsave("rim.png", rim)
        plt.imsave("centroid.png", centroid)
        plt.imsave("res_mean.png", stabilized_video.mean(dim=cold_starter.params.frames_axis))

        residuals = stabilized_video.where(rim > 0, 0)
        # Compute deviation from median
        V = cold_starter._center_to_median(residuals)

        # Compute energy (variance)
        E = (V**2).sum(dim="frame")

        # Find and analyze neighborhood of maximum variance
        neighborhood = cold_starter._get_max_energy_slice(arr=residuals, energy_landscape=E)
        a_merge, c_merge = cold_starter._local_nmf(
            neighborhood=neighborhood,
            spatial_sizes=residuals.sel({"frame": 0}).sizes,
            temporal_coords=residuals.coords,
        )

        valid = cold_starter._validate_new_components(
            new_footprints=a_merge,
            new_traces=c_merge,
            footprints=footprints,
            traces=traces,
            residuals=residuals,
        )

        assert valid
