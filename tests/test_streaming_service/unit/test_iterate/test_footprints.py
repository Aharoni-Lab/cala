from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import binary_erosion, binary_dilation

from cala.streaming.init.common.traces import TracesInitializer, TracesInitializerParams
from cala.streaming.init.odl.component_stats import (
    ComponentStatsInitializer,
    ComponentStatsInitializerParams,
)
from cala.streaming.init.odl.pixel_stats import (
    PixelStatsInitializer,
    PixelStatsInitializerParams,
)
from cala.streaming.iterate.footprints import FootprintsUpdater, FootprintsUpdaterParams


@dataclass
class TestParams:
    n_components = 5
    height = 10
    width = 10
    n_frames = 5


class TestFootprintUpdater:
    @pytest.fixture(scope="class")
    def updater(self):
        return FootprintsUpdater(FootprintsUpdaterParams(boundary_expansion_pixels=1))

    @pytest.fixture
    def p(self):
        return TestParams()

    @pytest.fixture
    def coords(self, p):
        return {
            "id_": ("component", [f"id{i}" for i in range(p.n_components)]),
            "type_": ("component", ["neuron"] * (p.n_components - 1) + ["background"]),
        }

    @pytest.fixture
    def sample_footprints(self, p, coords):
        """Create sample footprints with known overlap patterns."""
        footprints_data = np.zeros((p.n_components, p.height, p.width))
        # Set up specific overlap patterns
        footprints_data[0, 0:5, 0:5] = 1  # Component 0
        footprints_data[1, 3:8, 3:8] = 1  # Component 1 (overlaps with 0)
        footprints_data[2, 8:10, 8:10] = 1  # Component 2 (isolated)
        footprints_data[3, 0:3, 8:10] = 1  # Component 3
        footprints_data[4, 1:4, 7:9] = 1  # Component 4 (overlaps with 3)

        return xr.DataArray(
            footprints_data, dims=("component", "height", "width"), coords=coords
        )

    @pytest.fixture
    def sample_traces(self, p, coords):
        traces = xr.DataArray(
            np.zeros((p.n_components, p.n_frames)),
            dims=("component", "frame"),
            coords=coords,
        )
        traces[0, :] = [1 for _ in range(p.n_frames)]
        traces[1, :] = [i for i in range(p.n_frames)]
        traces[2, :] = [p.n_frames - 1 - i for i in range(p.n_frames)]
        traces[3, :] = [abs((p.n_frames - 1) / 2 - i) for i in range(p.n_frames)]
        traces[4, :] = np.random.rand(p.n_frames)
        return traces

    @pytest.fixture
    def sample_denoised(self, sample_footprints, sample_traces):
        return (sample_footprints @ sample_traces).transpose("frame", "height", "width")

    def get_stats(self, footprints, denoised):
        """Helper to compute traces and stats for modified footprints."""
        t_init = TracesInitializer(TracesInitializerParams())
        traces = t_init.learn_one(footprints, denoised).transform_one()

        ps = PixelStatsInitializer(PixelStatsInitializerParams())
        pixel_stats = ps.learn_one(traces=traces, frame=denoised).transform_one()

        cs = ComponentStatsInitializer(ComponentStatsInitializerParams())
        component_stats = cs.learn_one(traces=traces, frame=denoised).transform_one()

        return traces, pixel_stats, component_stats

    def visualize_iteration(
        self,
        visualizer,
        footprints,
        traces,
        pixel_stats,
        component_stats,
        sample_footprints,
        sample_denoised,
        subdir,
        name,
    ):
        """Helper to visualize iteration results."""
        # Plot initial state
        visualizer.plot_footprints(footprints, subdir=subdir, name=name)
        visualizer.plot_traces(traces, subdir=subdir)
        visualizer.plot_pixel_stats(pixel_stats, footprints, subdir=subdir)
        visualizer.plot_component_stats(component_stats, subdir=subdir)

        # Run updater and plot results
        updater = FootprintsUpdater(
            FootprintsUpdaterParams(boundary_expansion_pixels=1)
        )
        updater.learn_one(
            footprints=footprints,
            pixel_stats=pixel_stats,
            component_stats=component_stats,
        )
        new_footprints = updater.transform_one().transpose(*sample_footprints.dims)

        visualizer.plot_footprints(new_footprints, subdir=subdir, name="pred")
        visualizer.plot_comparison(
            sample_footprints.max(dim="component"),
            new_footprints.max(dim="component"),
            subdir=subdir,
        )

        # Visualize movies
        preconstructed_movie = (footprints @ traces).transpose(*sample_denoised.dims)
        postconstructed_movie = (new_footprints @ traces).transpose(
            *sample_denoised.dims
        )
        residual = sample_denoised - postconstructed_movie

        visualizer.save_video_frames(
            [
                (sample_denoised, "label"),
                (preconstructed_movie, "preconstructed"),
                (postconstructed_movie, "postconstructed"),
                (residual, "residual"),
            ],
            subdir=subdir,
            name="recovered_movie",
        )

        return new_footprints

    @pytest.mark.viz
    def test_perfect_condition(
        self,
        visualizer,
        sample_footprints,
        sample_traces,
        sample_denoised,
    ):
        ps = PixelStatsInitializer(PixelStatsInitializerParams())
        sample_pixel_stats = ps.learn_one(
            traces=sample_traces, frame=sample_denoised
        ).transform_one()

        cs = ComponentStatsInitializer(ComponentStatsInitializerParams())
        sample_component_stats = cs.learn_one(
            traces=sample_traces, frame=sample_denoised
        ).transform_one()

        new_footprints = self.visualize_iteration(
            visualizer,
            sample_footprints,
            sample_traces,
            sample_pixel_stats,
            sample_component_stats,
            sample_footprints,
            sample_denoised,
            subdir="iter/footprints",
            name="label",
        )
        assert np.allclose(
            new_footprints, sample_footprints.transpose(*new_footprints.dims), atol=1e-3
        )

    @pytest.mark.viz
    def test_imperfect_condition(self, visualizer, sample_footprints, sample_denoised):
        # Add noise to stats
        traces, pixel_stats, component_stats = self.get_stats(
            sample_footprints, sample_denoised
        )
        noisy_pixel_stats = pixel_stats + 0.1 * np.random.rand(*pixel_stats.shape)
        noisy_component_stats = component_stats + 0.1 * np.random.rand(
            *component_stats.shape
        )

        self.visualize_iteration(
            visualizer,
            sample_footprints,
            traces,
            noisy_pixel_stats,
            noisy_component_stats,
            sample_footprints,
            sample_denoised,
            subdir="iter/footprints/imperfect",
            name="label",
        )

    @pytest.mark.viz
    def test_wrong_footprint(self, visualizer, sample_footprints, sample_denoised):
        wrong_footprints = sample_footprints.copy()[:4]
        wrong_footprints[3] = sample_footprints[3] + sample_footprints[4]

        traces, pixel_stats, component_stats = self.get_stats(
            wrong_footprints, sample_denoised
        )

        self.visualize_iteration(
            visualizer,
            wrong_footprints,
            traces,
            pixel_stats,
            component_stats,
            sample_footprints,
            sample_denoised,
            subdir="iter/footprints/wrong",
            name="wrong",
        )

    @pytest.mark.viz
    def test_small_footprint(self, visualizer, sample_footprints, sample_denoised):
        small_footprints = sample_footprints.copy()
        small_footprints[1] = binary_erosion(small_footprints[1])

        traces, pixel_stats, component_stats = self.get_stats(
            small_footprints, sample_denoised
        )

        self.visualize_iteration(
            visualizer,
            small_footprints,
            traces,
            pixel_stats,
            component_stats,
            sample_footprints,
            sample_denoised,
            subdir="iter/footprints/small",
            name="small",
        )

    @pytest.mark.viz
    def test_oversized_footprint(self, visualizer, sample_footprints, sample_denoised):
        oversized_footprints = sample_footprints.copy()
        oversized_footprints[1] = binary_dilation(oversized_footprints[1])

        traces, pixel_stats, component_stats = self.get_stats(
            oversized_footprints, sample_denoised
        )

        self.visualize_iteration(
            visualizer,
            oversized_footprints,
            traces,
            pixel_stats,
            component_stats,
            sample_footprints,
            sample_denoised,
            subdir="iter/footprints/oversized",
            name="oversized",
        )

    @pytest.mark.viz
    def test_redundant_footprint(self, visualizer, sample_footprints, sample_denoised):
        redundant_footprints = sample_footprints.copy()
        rolled = xr.DataArray(
            np.roll(sample_footprints[-1], -1), dims=("height", "width")
        )
        rolled = rolled.expand_dims("component").assign_coords(
            {"id_": ("component", ["id5"]), "type_": ("component", ["neuron"])}
        )
        redundant_footprints = xr.concat(
            [redundant_footprints, rolled],
            dim="component",
        )

        traces, pixel_stats, component_stats = self.get_stats(
            redundant_footprints, sample_denoised
        )

        self.visualize_iteration(
            visualizer,
            redundant_footprints,
            traces,
            pixel_stats,
            component_stats,
            sample_footprints,
            sample_denoised,
            subdir="iter/footprints/redundant",
            name="redundant",
        )
