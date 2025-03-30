from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

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
    """need to simulate:
    footprints: Footprints,
    pixel_stats: PixelStats,
    component_stats: ComponentStats,
    frame: Frame,
    """

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
        """Create sample footprints for testing.

        Creates a set of footprints with known overlap patterns:
        - Components 0 and 1 overlap
        - Component 2 is isolated
        - Components 3 and 4 overlap
        """
        # Create empty footprints
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
    def sample_residuals(self, p):
        residual = xr.DataArray(
            np.zeros((p.n_frames, p.height, p.width)), dims=("frame", "height", "width")
        )
        for i in range(p.n_frames):
            residual[i, :, i % p.width] = 3

        return residual

    @pytest.fixture
    def sample_denoised(self, sample_footprints, sample_traces):
        return (sample_footprints @ sample_traces).transpose("frame", "height", "width")

    @pytest.fixture
    def sample_pixel_stats(self, sample_denoised, sample_traces):
        ps = PixelStatsInitializer(PixelStatsInitializerParams())
        ps.learn_one(traces=sample_traces, frame=sample_denoised)
        return ps.transform_one()

    @pytest.fixture
    def sample_component_stats(self, sample_denoised, sample_traces):
        cs = ComponentStatsInitializer(ComponentStatsInitializerParams())
        cs.learn_one(traces=sample_traces, frame=sample_denoised)
        return cs.transform_one()

    @pytest.mark.viz
    def test_perfect_condition(
        self,
        updater,
        visualizer,
        sample_footprints,
        sample_traces,
        sample_pixel_stats,
        sample_component_stats,
        sample_denoised,
    ):
        visualizer.plot_footprints(
            sample_footprints, subdir="iter/footprints", name="label"
        )
        visualizer.plot_traces(sample_traces, subdir="iter/footprints")
        visualizer.plot_trace_correlations(sample_traces, subdir="iter/footprints")
        visualizer.save_video_frames(sample_denoised, subdir="iter/footprints")
        visualizer.plot_pixel_stats(
            sample_pixel_stats, sample_footprints, subdir="iter/footprints"
        )
        visualizer.plot_component_stats(
            sample_component_stats, subdir="iter/footprints"
        )

        updater.learn_one(
            footprints=sample_footprints,
            pixel_stats=sample_pixel_stats,
            component_stats=sample_component_stats,
        )
        new_footprints = updater.transform_one()

        visualizer.plot_footprints(
            new_footprints, subdir="iter/footprints", name="pred"
        )
        visualizer.plot_comparison(
            sample_footprints.max(dim="component"),
            new_footprints.max(dim="component"),
            subdir="iter/footprints",
        )

    @pytest.mark.viz
    def test_imperfect_condition(
        self,
        updater,
        visualizer,
        sample_footprints,
        sample_traces,
        sample_pixel_stats,
        sample_component_stats,
        sample_denoised,
    ):
        sample_pixel_stats = sample_pixel_stats + 0.1 * np.random.rand(
            *sample_pixel_stats.shape
        )
        sample_component_stats = sample_component_stats + 0.1 * np.random.rand(
            *sample_component_stats.shape
        )

        visualizer.plot_footprints(
            sample_footprints, subdir="iter/footprints/imperfect", name="label"
        )
        visualizer.plot_pixel_stats(
            sample_pixel_stats, sample_footprints, subdir="iter/footprints/imperfect"
        )
        visualizer.plot_component_stats(
            sample_component_stats, subdir="iter/footprints/imperfect"
        )

        updater.learn_one(
            footprints=sample_footprints,
            pixel_stats=sample_pixel_stats,
            component_stats=sample_component_stats,
        )
        new_footprints = updater.transform_one()

        visualizer.plot_footprints(
            new_footprints, subdir="iter/footprints/imperfect", name="pred"
        )
        visualizer.plot_comparison(
            sample_footprints.max(dim="component"),
            new_footprints.max(dim="component"),
            subdir="iter/footprints/imperfect",
        )

    @pytest.mark.viz
    def test_wrong_footprint(
        self,
        updater,
        visualizer,
        sample_footprints,
        sample_denoised,
    ):
        wrong_footprints = sample_footprints.copy()[:4]
        wrong_footprints[3] = sample_footprints[3] + sample_footprints[4]

        t_init = TracesInitializer(TracesInitializerParams())
        wrong_traces = t_init.learn_one(
            wrong_footprints, sample_denoised
        ).transform_one()

        ps = PixelStatsInitializer(PixelStatsInitializerParams())
        wrong_pixel_stats = ps.learn_one(
            traces=wrong_traces, frame=sample_denoised
        ).transform_one()

        cs = ComponentStatsInitializer(ComponentStatsInitializerParams())
        wrong_component_stats = cs.learn_one(
            traces=wrong_traces, frame=sample_denoised
        ).transform_one()

        visualizer.plot_footprints(
            wrong_footprints, subdir="iter/footprints/wrong", name="wrong"
        )
        visualizer.plot_traces(wrong_traces, subdir="iter/footprints/wrong")
        visualizer.plot_pixel_stats(
            wrong_pixel_stats, wrong_footprints, subdir="iter/footprints/wrong"
        )
        visualizer.plot_component_stats(
            wrong_component_stats, subdir="iter/footprints/wrong"
        )

        updater.learn_one(
            footprints=wrong_footprints,
            pixel_stats=wrong_pixel_stats,
            component_stats=wrong_component_stats,
        )
        new_footprints = updater.transform_one()

        visualizer.plot_footprints(
            new_footprints, subdir="iter/footprints/wrong", name="pred"
        )
        visualizer.plot_comparison(
            sample_footprints.max(dim="component"),
            new_footprints.max(dim="component"),
            subdir="iter/footprints/wrong",
        )

        preconstructed_movie = (wrong_footprints @ wrong_traces).transpose(
            *sample_denoised.dims
        )
        postconstructed_movie = (new_footprints @ wrong_traces).transpose(
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
            subdir="iter/footprints/wrong",
            name="recovered_movie",
        )

    @pytest.mark.viz
    def test_small_footprint(
        self,
        updater,
        visualizer,
        sample_footprints,
        sample_denoised,
    ):
        from scipy.ndimage import binary_erosion

        small_footprints = sample_footprints.copy()
        small_footprints[1] = binary_erosion(small_footprints[1])
        t_init = TracesInitializer(TracesInitializerParams())
        small_traces = t_init.learn_one(
            small_footprints, sample_denoised
        ).transform_one()

        ps = PixelStatsInitializer(PixelStatsInitializerParams())
        small_pixel_stats = ps.learn_one(
            traces=small_traces, frame=sample_denoised
        ).transform_one()

        cs = ComponentStatsInitializer(ComponentStatsInitializerParams())
        small_component_stats = cs.learn_one(
            traces=small_traces, frame=sample_denoised
        ).transform_one()

        visualizer.plot_footprints(
            small_footprints, subdir="iter/footprints/small", name="small"
        )
        visualizer.plot_traces(small_traces, subdir="iter/footprints/small")
        visualizer.plot_pixel_stats(
            small_pixel_stats, small_footprints, subdir="iter/footprints/small"
        )
        visualizer.plot_component_stats(
            small_component_stats, subdir="iter/footprints/small"
        )

        updater.learn_one(
            footprints=small_footprints,
            pixel_stats=small_pixel_stats,
            component_stats=small_component_stats,
        )
        new_footprints = updater.transform_one().transpose(*sample_footprints.dims)

        visualizer.plot_footprints(
            new_footprints, subdir="iter/footprints/small", name="pred"
        )
        visualizer.plot_comparison(
            sample_footprints.max(dim="component"),
            new_footprints.max(dim="component"),
            subdir="iter/footprints/small",
        )

        preconstructed_movie = (small_footprints @ small_traces).transpose(
            *sample_denoised.dims
        )
        postconstructed_movie = (new_footprints @ small_traces).transpose(
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
            subdir="iter/footprints/small",
            name="recovered_movie",
        )

    @pytest.mark.viz
    def test_oversized_footprint(
        self,
        updater,
        visualizer,
        sample_footprints,
        sample_denoised,
    ):
        from scipy.ndimage import binary_dilation

        oversized_footprints = sample_footprints.copy()
        oversized_footprints[1] = binary_dilation(oversized_footprints[1])
        t_init = TracesInitializer(TracesInitializerParams())
        oversized_traces = t_init.learn_one(
            oversized_footprints, sample_denoised
        ).transform_one()

        ps = PixelStatsInitializer(PixelStatsInitializerParams())
        oversized_pixel_stats = ps.learn_one(
            traces=oversized_traces, frame=sample_denoised
        ).transform_one()

        cs = ComponentStatsInitializer(ComponentStatsInitializerParams())
        oversized_component_stats = cs.learn_one(
            traces=oversized_traces, frame=sample_denoised
        ).transform_one()

        visualizer.plot_footprints(
            oversized_footprints, subdir="iter/footprints/oversized", name="oversized"
        )
        visualizer.plot_traces(oversized_traces, subdir="iter/footprints/oversized")
        visualizer.plot_pixel_stats(
            oversized_pixel_stats,
            oversized_footprints,
            subdir="iter/footprints/oversized",
        )
        visualizer.plot_component_stats(
            oversized_component_stats, subdir="iter/footprints/oversized"
        )

        updater.learn_one(
            footprints=oversized_footprints,
            pixel_stats=oversized_pixel_stats,
            component_stats=oversized_component_stats,
        )
        new_footprints = updater.transform_one().transpose(*sample_footprints.dims)

        visualizer.plot_footprints(
            new_footprints, subdir="iter/footprints/oversized", name="pred"
        )
        visualizer.plot_comparison(
            sample_footprints.max(dim="component"),
            new_footprints.max(dim="component"),
            subdir="iter/footprints/oversized",
        )

        preconstructed_movie = (oversized_footprints @ oversized_traces).transpose(
            *sample_denoised.dims
        )
        postconstructed_movie = (new_footprints @ oversized_traces).transpose(
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
            subdir="iter/footprints/oversized",
            name="recovered_movie",
        )
