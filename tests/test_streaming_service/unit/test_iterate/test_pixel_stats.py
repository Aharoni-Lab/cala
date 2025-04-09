import numpy as np
import pytest

from cala.streaming.nodes.init.odl.pixel_stats import (
    PixelStatsInitializer,
    PixelStatsInitializerParams,
)
from cala.streaming.nodes.iter.pixel_stats import (
    PixelStatsUpdater,
    PixelStatsUpdaterParams,
)
from cala.streaming.util import package_frame


class TestPixelStatsUpdater:
    """need to simulate:
    frame: Frame,
    traces: Traces,
    component_stats: ComponentStats,
    """

    @pytest.fixture(scope="class")
    def updater(self):
        return PixelStatsUpdater(PixelStatsUpdaterParams())

    @pytest.fixture(scope="class")
    def initializer(self):
        return PixelStatsInitializer(PixelStatsInitializerParams())

    @pytest.fixture
    def prev_pixel_stats(self, initializer, mini_traces, mini_denoised):
        """this should look like it was last update before the current frame.
        (so before the most recent frame index in mini_traces)
        """
        traces_to_use = mini_traces.isel(frame=slice(None, -1))
        frames_to_use = mini_denoised.isel(frame=slice(None, -1))

        # doesn't matter we're only using it for the frame count
        initializer.learn_one(traces=traces_to_use, frame=frames_to_use)
        return initializer.transform_one()

    @pytest.mark.viz
    def test_sanity_check(
        self,
        visualizer,
        updater,
        mini_footprints,
        mini_traces,
        prev_pixel_stats,
        mini_denoised,
        initializer,
    ):
        visualizer.plot_footprints(mini_footprints, subdir="iter/pixel_stats")
        visualizer.plot_traces(mini_traces, subdir="iter/pixel_stats")
        visualizer.plot_trace_correlations(mini_traces, subdir="iter/pixel_stats")
        visualizer.save_video_frames(mini_denoised, subdir="iter/pixel_stats")
        visualizer.plot_pixel_stats(
            prev_pixel_stats.transpose(*mini_footprints.dims),
            mini_footprints,
            subdir="iter/pixel_stats",
            name="prev_ps",
        )
        updater.learn_one(
            frame=package_frame(mini_denoised[-1].values, len(mini_denoised) - 1),
            traces=mini_traces,
            pixel_stats=prev_pixel_stats,
        )
        new_pixel_stats = updater.transform_one()
        visualizer.plot_pixel_stats(
            new_pixel_stats.transpose(*mini_footprints.dims),
            mini_footprints,
            subdir="iter/pixel_stats",
            name="new_ps",
        )

        late_init_ps = initializer.learn_one(
            mini_traces,
            frame=mini_denoised,
        ).transform_one()

        visualizer.plot_pixel_stats(
            late_init_ps.transpose(*mini_footprints.dims),
            mini_footprints,
            subdir="iter/pixel_stats",
            name="late_ps",
        )

        assert np.allclose(
            new_pixel_stats.transpose(*mini_footprints.dims),
            late_init_ps.transpose(*mini_footprints.dims),
        )
