from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from cala.streaming.composer import Frame
from cala.streaming.init.odl.component_stats import (
    ComponentStatsInitializer,
    ComponentStatsInitializerParams,
)
from cala.streaming.iterate.component_stats import (
    ComponentStatsUpdater,
    ComponentStatsUpdaterParams,
)


@dataclass
class TestParams:
    n_components = 5
    height = 10
    width = 10
    n_frames = 5


class TestCompStatsUpdater:
    """need to simulate:
    frame: Frame,
    traces: Traces,
    component_stats: ComponentStats,
    """

    @pytest.fixture(scope="class")
    def updater(self):
        return ComponentStatsUpdater(ComponentStatsUpdaterParams())

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
        return ComponentStatsInitializer(ComponentStatsInitializerParams())

    @pytest.fixture
    def prev_comp_stats(self, initializer, sample_traces, p):
        """this should look like it was last update before the current frame.
        (so before the most recent frame index in sample_traces)
        """
        traces_to_use = sample_traces.isel(frame=slice(None, -1))

        # doesn't matter we're only using it for the frame count
        initializer.learn_one(traces=traces_to_use, frame=traces_to_use)
        return initializer.transform_one()

    @pytest.mark.viz
    def test_sanity_check(
        self,
        visualizer,
        updater,
        sample_traces,
        prev_comp_stats,
        sample_denoised,
    ):
        visualizer.plot_traces(sample_traces, subdir="iter/comp_stats")
        visualizer.plot_trace_correlations(sample_traces, subdir="iter/comp_stats")
        visualizer.save_video_frames(sample_denoised, subdir="iter/comp_stats")
        visualizer.plot_component_stats(
            prev_comp_stats, subdir="iter/comp_stats", name="prev_cs"
        )
        updater.learn_one(
            frame=Frame(sample_denoised[-1], len(sample_denoised)),
            traces=sample_traces,
            component_stats=prev_comp_stats,
        )
        new_comp_stats = updater.transform_one()
        visualizer.plot_component_stats(
            new_comp_stats, subdir="iter/comp_stats", name="new_cs"
        )
