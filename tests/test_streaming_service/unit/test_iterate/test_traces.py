import numpy as np
import pytest

from cala.streaming.composer import Frame
from cala.streaming.init.odl.overlaps import (
    OverlapsInitializer,
    OverlapsInitializerParams,
)
from cala.streaming.iterate.traces import TracesUpdater, TracesUpdaterParams


class TestTraceUpdater:
    """need to simulate:
    footprints: Footprints,
    traces: Traces,
    frame: Frame,
    overlaps: Overlaps
    """

    @pytest.fixture(scope="class")
    def updater(self):
        return TracesUpdater(TracesUpdaterParams(tolerance=1e-3))

    @pytest.fixture
    def mini_overlap(self, mini_footprints):
        overlapper = OverlapsInitializer(OverlapsInitializerParams())

        overlap = overlapper.learn_one(mini_footprints).transform_one()
        overlap.values = overlap.data.todense()
        return overlap

    @pytest.mark.viz
    def test_sanity_check(
        self,
        mini_params,
        updater,
        mini_footprints,
        mini_traces,
        mini_overlap,
        mini_denoised,
        visualizer,
    ):
        visualizer.plot_footprints(mini_footprints, subdir="iter/trace")
        visualizer.plot_traces(mini_traces, subdir="iter/trace")
        visualizer.save_video_frames(mini_denoised, subdir="iter/trace")
        visualizer.plot_overlaps(
            mini_overlap, footprints=mini_footprints, subdir="iter/trace"
        )
        updater.learn_one(
            footprints=mini_footprints,
            traces=mini_traces.isel(frame=slice(None, -1)),
            overlaps=mini_overlap,
            frame=Frame(mini_denoised[-1], mini_params.n_frames - 1),
        )
        new_traces = updater.transform_one()

        visualizer.plot_comparison(
            mini_footprints @ new_traces,
            mini_footprints @ mini_traces.isel(frame=-1),
            subdir="iter/trace",
        )

        assert np.allclose(
            new_traces, mini_traces.isel(frame=-1), atol=1e-3 * mini_params.n_components
        )
