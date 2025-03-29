from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from cala.streaming.composer import Frame
from cala.streaming.init.odl.pixel_stats import (
    PixelStatsInitializer,
    PixelStatsInitializerParams,
)
from cala.streaming.iterate.pixel_stats import (
    PixelStatsUpdater,
    PixelStatsUpdaterParams,
)


@dataclass
class TestParams:
    n_components = 5
    height = 10
    width = 10
    n_frames = 5


class TestPixelStatsUpdater:
    """need to simulate:
    frame: Frame,
    traces: Traces,
    component_stats: ComponentStats,
    """

    @pytest.fixture(scope="class")
    def updater(self):
        return PixelStatsUpdater(PixelStatsUpdaterParams())

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
    def sample_movie(self, sample_denoised, sample_residuals):
        return (sample_denoised + sample_residuals).transpose(
            "frame", "height", "width"
        )

    @pytest.fixture(scope="class")
    def initializer(self):
        return PixelStatsInitializer(PixelStatsInitializerParams())

    @pytest.fixture
    def prev_pixel_stats(self, initializer, sample_traces, sample_denoised):
        """this should look like it was last update before the current frame.
        (so before the most recent frame index in sample_traces)
        """
        traces_to_use = sample_traces.isel(frame=slice(None, -1))
        frames_to_use = sample_denoised.isel(frame=slice(None, -1))

        # doesn't matter we're only using it for the frame count
        initializer.learn_one(traces=traces_to_use, frame=frames_to_use)
        return initializer.transform_one()

    @pytest.mark.viz
    def test_sanity_check(
        self,
        visualizer,
        updater,
        sample_footprints,
        sample_traces,
        prev_pixel_stats,
        sample_denoised,
        initializer,
    ):
        visualizer.plot_footprints(sample_footprints, subdir="iter/pixel_stats")
        visualizer.plot_traces(sample_traces, subdir="iter/pixel_stats")
        visualizer.plot_trace_correlations(sample_traces, subdir="iter/pixel_stats")
        visualizer.save_video_frames(sample_denoised, subdir="iter/pixel_stats")
        visualizer.plot_pixel_stats(
            prev_pixel_stats,
            sample_footprints,
            subdir="iter/pixel_stats",
            name="prev_ps",
        )
        updater.learn_one(
            frame=Frame(sample_denoised[-1], len(sample_denoised)),
            traces=sample_traces,
            pixel_stats=prev_pixel_stats,
        )
        new_pixel_stats = updater.transform_one()
        visualizer.plot_pixel_stats(
            new_pixel_stats, sample_footprints, subdir="iter/pixel_stats", name="new_ps"
        )

        late_init_ps = initializer.learn_one(
            sample_traces,
            frame=sample_denoised,
        ).transform_one()

        visualizer.plot_pixel_stats(
            late_init_ps, sample_footprints, subdir="iter/pixel_stats", name="late_ps"
        )
