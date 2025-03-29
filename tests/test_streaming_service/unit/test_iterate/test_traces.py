from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from cala.streaming.composer import Frame
from cala.streaming.init.odl.overlaps import (
    OverlapsInitializer,
    OverlapsInitializerParams,
)
from cala.streaming.iterate.traces import TracesUpdater, TracesUpdaterParams


@dataclass
class TestTraceUpdaterParams:
    n_components = 5
    height = 10
    width = 10
    n_frames = 5


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
    def p(self):
        return TestTraceUpdaterParams()

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
        traces[2, :] = [p.n_frames - i for i in range(p.n_frames)]
        traces[3, :] = [abs(p.n_frames / 2 - i) for i in range(p.n_frames)]
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

    @pytest.fixture
    def sample_overlap(self, sample_footprints):
        overlapper = OverlapsInitializer(OverlapsInitializerParams())

        overlap = overlapper.learn_one(sample_footprints).transform_one()
        overlap.values = overlap.data.todense()
        return overlap

    @pytest.mark.viz
    def test_sanity_check(
        self,
        p,
        updater,
        sample_footprints,
        sample_traces,
        sample_overlap,
        sample_denoised,
        visualizer,
    ):
        visualizer.plot_footprints(sample_footprints, subdir="iter/trace")
        visualizer.plot_traces(sample_traces, subdir="iter/trace")
        visualizer.save_video_frames(sample_denoised, subdir="iter/trace")
        visualizer.plot_overlap(
            sample_overlap, footprints=sample_footprints, subdir="iter/trace"
        )
        updater.learn_one(
            footprints=sample_footprints,
            traces=sample_traces.isel(frame=slice(None, -1)),
            overlaps=sample_overlap,
            frame=Frame(sample_denoised[-1], p.n_frames),
        )
        new_traces = updater.transform_one()

        assert np.allclose(
            new_traces, sample_traces.isel(frame=-1), atol=1e-3 * p.n_components
        )
