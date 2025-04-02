import numpy as np
import pytest

from cala.streaming.composer import Frame
from cala.streaming.init.odl.component_stats import (
    ComponentStatsInitializer,
    ComponentStatsInitializerParams,
)
from cala.streaming.iterate.component_stats import (
    ComponentStatsUpdater,
    ComponentStatsUpdaterParams,
)


class TestCompStatsUpdater:
    """need to simulate:
    frame: Frame,
    traces: Traces,
    component_stats: ComponentStats,
    """

    @pytest.fixture(scope="class")
    def updater(self):
        return ComponentStatsUpdater(ComponentStatsUpdaterParams())

    @pytest.fixture(scope="class")
    def initializer(self):
        return ComponentStatsInitializer(ComponentStatsInitializerParams())

    @pytest.fixture
    def prev_comp_stats(self, initializer, mini_traces, mini_params):
        """this should look like it was last update before the current frame.
        (so before the most recent frame index in mini_traces)
        """
        traces_to_use = mini_traces.isel(frame=slice(None, -1))

        # doesn't matter we're only using it for the frame count
        initializer.learn_one(traces=traces_to_use, frame=traces_to_use)
        return initializer.transform_one()

    @pytest.mark.viz
    def test_sanity_check(
        self,
        visualizer,
        updater,
        mini_footprints,
        mini_traces,
        prev_comp_stats,
        mini_denoised,
        initializer,
    ):
        visualizer.plot_footprints(mini_footprints, subdir="iter/comp_stats")
        visualizer.plot_traces(mini_traces, subdir="iter/comp_stats")
        visualizer.plot_trace_correlations(mini_traces, subdir="iter/comp_stats")
        visualizer.save_video_frames(mini_denoised, subdir="iter/comp_stats")
        visualizer.plot_component_stats(
            prev_comp_stats, subdir="iter/comp_stats", name="prev_cs"
        )
        updater.learn_one(
            frame=Frame(mini_denoised[-1], len(mini_denoised) - 1),
            traces=mini_traces,
            component_stats=prev_comp_stats,
        )
        new_comp_stats = updater.transform_one()
        visualizer.plot_component_stats(
            new_comp_stats, subdir="iter/comp_stats", name="new_cs"
        )

        late_init_cs = initializer.learn_one(
            mini_traces,
            frame=mini_denoised,
        ).transform_one()

        visualizer.plot_component_stats(
            late_init_cs, subdir="iter/comp_stats", name="late_cs"
        )

        assert np.allclose(late_init_cs, new_comp_stats)
