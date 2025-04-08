from typing import Any

import numpy as np
import pytest
import xarray as xr

from cala.streaming.nodes.init.odl.component_stats import (
    ComponentStatsInitializer,
    ComponentStatsInitializerParams,
)
from cala.streaming.nodes.iter.component_stats import (
    ComponentStatsUpdater,
    ComponentStatsUpdaterParams,
)
from cala.streaming.util import package_frame
from cala.viz_util import Visualizer


class TestCompStatsUpdater:
    """need to simulate:
    frame: Frame,
    traces: Traces,
    component_stats: ComponentStats,
    """

    @pytest.fixture(scope="class")
    def updater(self) -> ComponentStatsUpdater:
        return ComponentStatsUpdater(ComponentStatsUpdaterParams())

    @pytest.fixture(scope="class")
    def initializer(self) -> ComponentStatsInitializer:
        return ComponentStatsInitializer(ComponentStatsInitializerParams())

    @pytest.fixture
    def prev_comp_stats(
        self, initializer: ComponentStatsInitializer, mini_traces: xr.DataArray, mini_params: Any
    ) -> xr.DataArray:
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
        visualizer: Visualizer,
        updater: ComponentStatsUpdater,
        mini_footprints: xr.DataArray,
        mini_traces: xr.DataArray,
        prev_comp_stats: xr.DataArray,
        mini_denoised: xr.DataArray,
        initializer: ComponentStatsInitializer,
    ) -> None:
        visualizer.plot_footprints(mini_footprints, subdir="iter/comp_stats")
        visualizer.plot_traces(mini_traces, subdir="iter/comp_stats")
        visualizer.plot_trace_correlations(mini_traces, subdir="iter/comp_stats")
        visualizer.save_video_frames(mini_denoised, subdir="iter/comp_stats")
        visualizer.plot_component_stats(prev_comp_stats, subdir="iter/comp_stats", name="prev_cs")
        updater.learn_one(
            frame=package_frame(mini_denoised[-1].values, len(mini_denoised) - 1),
            traces=mini_traces,
            component_stats=prev_comp_stats,
        )
        new_comp_stats = updater.transform_one()
        visualizer.plot_component_stats(new_comp_stats, subdir="iter/comp_stats", name="new_cs")

        late_init_cs = initializer.learn_one(
            mini_traces,
            frame=mini_denoised,
        ).transform_one()

        visualizer.plot_component_stats(late_init_cs, subdir="iter/comp_stats", name="late_cs")

        assert np.allclose(late_init_cs, new_comp_stats)
