from typing import Any

import numpy as np
import pytest
import xarray as xr

from cala.gui.plots import Plotter
from cala.nodes.init.odl.overlaps import OverlapsInitializer, OverlapsInitializerParams
from cala.nodes.iter.traces import TracesUpdater, TracesUpdaterParams
from cala.util.new import package_frame


class TestTraceUpdater:
    """need to simulate:
    footprints: Footprints,
    traces: Traces,
    frame: Frame,
    overlaps: Overlaps
    """

    @pytest.fixture(scope="class")
    def updater(self) -> TracesUpdater:
        return TracesUpdater(TracesUpdaterParams(tolerance=1e-3))

    @pytest.fixture
    def mini_overlap(self, mini_footprints: xr.DataArray) -> xr.DataArray:
        overlapper = OverlapsInitializer(OverlapsInitializerParams())

        overlap = overlapper.learn_one(mini_footprints).transform_one()
        overlap.values = overlap.data.todense()
        return overlap

    @pytest.mark.viz
    def test_sanity_check(
        self,
        mini_params: Any,
        updater: TracesUpdater,
        mini_footprints: xr.DataArray,
        mini_traces: xr.DataArray,
        mini_overlap: xr.DataArray,
        mini_denoised: xr.DataArray,
        plotter: Plotter,
    ) -> None:
        plotter.plot_footprints(mini_footprints, subdir="iter/trace")
        plotter.plot_traces(mini_traces, subdir="iter/trace")
        plotter.save_video_frames(mini_denoised, subdir="iter/trace")
        plotter.plot_overlaps(mini_overlap, footprints=mini_footprints, subdir="iter/trace")
        updater.learn_one(
            footprints=mini_footprints,
            traces=mini_traces.isel(frame=slice(None, -1)),
            overlaps=mini_overlap,
            frame=package_frame(
                mini_denoised[-1].values,
                mini_params.n_frames - 1,
                mini_denoised[-1].coords["time_"].item(),
            ),
        )
        new_traces = updater.transform_one()

        plotter.plot_comparison(
            mini_footprints @ new_traces,
            mini_footprints @ mini_traces.isel(frame=-1),
            subdir="iter/trace",
        )

        assert np.allclose(
            new_traces, mini_traces.isel(frame=[-1]), atol=1e-3 * mini_params.n_components
        )
