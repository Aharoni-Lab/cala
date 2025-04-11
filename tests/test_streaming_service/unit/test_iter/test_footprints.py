import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import binary_dilation, binary_erosion

from cala.streaming.core import Component
from cala.streaming.nodes.init.common import (
    TracesInitializer,
    TracesInitializerParams,
)
from cala.streaming.nodes.init.odl.component_stats import (
    ComponentStatsInitializer,
    ComponentStatsInitializerParams,
)
from cala.streaming.nodes.init.odl.pixel_stats import (
    PixelStatsInitializer,
    PixelStatsInitializerParams,
)
from cala.streaming.nodes.iter.footprints import FootprintsUpdater, FootprintsUpdaterParams
from cala.viz_util import Visualizer


class TestFootprintUpdater:
    @pytest.fixture(scope="class")
    def updater(self) -> FootprintsUpdater:
        return FootprintsUpdater(FootprintsUpdaterParams(boundary_expansion_pixels=1))

    def get_stats(
        self, footprints: xr.DataArray, denoised: xr.DataArray
    ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
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
        visualizer: Visualizer,
        footprints: xr.DataArray,
        traces: xr.DataArray,
        pixel_stats: xr.DataArray,
        component_stats: xr.DataArray,
        mini_footprints: xr.DataArray,
        mini_denoised: xr.DataArray,
        subdir: str,
        name: str,
    ) -> xr.DataArray:
        """Helper to visualize iteration results."""
        # Plot initial state
        visualizer.plot_footprints(footprints, subdir=subdir, name=name)
        visualizer.plot_traces(traces, subdir=subdir)
        visualizer.plot_pixel_stats(
            pixel_stats.transpose(*footprints.dims), footprints, subdir=subdir
        )
        visualizer.plot_component_stats(component_stats, subdir=subdir)

        # Run updater and plot results
        updater = FootprintsUpdater(FootprintsUpdaterParams(boundary_expansion_pixels=1))
        updater.learn_one(
            footprints=footprints,
            pixel_stats=pixel_stats,
            component_stats=component_stats,
        )
        new_footprints = updater.transform_one().transpose(*mini_footprints.dims)

        visualizer.plot_footprints(new_footprints, subdir=subdir, name="pred")
        visualizer.plot_comparison(
            mini_footprints.max(dim="component"),
            new_footprints.max(dim="component"),
            subdir=subdir,
        )

        # Visualize movies
        preconstructed_movie = (footprints @ traces).transpose(*mini_denoised.dims)
        postconstructed_movie = (new_footprints @ traces).transpose(*mini_denoised.dims)
        residual = mini_denoised - postconstructed_movie

        visualizer.save_video_frames(
            [
                (mini_denoised, "label"),
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
        visualizer: Visualizer,
        mini_footprints: xr.DataArray,
        mini_traces: xr.DataArray,
        mini_denoised: xr.DataArray,
    ) -> None:
        ps = PixelStatsInitializer(PixelStatsInitializerParams())
        mini_pixel_stats = ps.learn_one(traces=mini_traces, frame=mini_denoised).transform_one()

        cs = ComponentStatsInitializer(ComponentStatsInitializerParams())
        mini_component_stats = cs.learn_one(traces=mini_traces, frame=mini_denoised).transform_one()

        new_footprints = self.visualize_iteration(
            visualizer,
            mini_footprints,
            mini_traces,
            mini_pixel_stats,
            mini_component_stats,
            mini_footprints,
            mini_denoised,
            subdir="iter/footprints",
            name="label",
        )
        assert np.allclose(
            new_footprints, mini_footprints.transpose(*new_footprints.dims), atol=1e-3
        )

    @pytest.mark.viz
    def test_imperfect_condition(
        self, visualizer: Visualizer, mini_footprints: xr.DataArray, mini_denoised: xr.DataArray
    ) -> None:
        # Add noise to stats
        traces, pixel_stats, component_stats = self.get_stats(mini_footprints, mini_denoised)
        noisy_pixel_stats = pixel_stats + 0.1 * np.random.rand(*pixel_stats.shape)
        noisy_component_stats = component_stats + 0.1 * np.random.rand(*component_stats.shape)

        self.visualize_iteration(
            visualizer,
            mini_footprints,
            traces,
            noisy_pixel_stats,
            noisy_component_stats,
            mini_footprints,
            mini_denoised,
            subdir="iter/footprints/imperfect",
            name="label",
        )

    @pytest.mark.viz
    def test_wrong_footprint(
        self, visualizer: Visualizer, mini_footprints: xr.DataArray, mini_denoised: xr.DataArray
    ) -> None:
        wrong_footprints = mini_footprints.copy()[:4]
        wrong_footprints[3] = mini_footprints[3] + mini_footprints[4]

        traces, pixel_stats, component_stats = self.get_stats(wrong_footprints, mini_denoised)

        self.visualize_iteration(
            visualizer,
            wrong_footprints,
            traces,
            pixel_stats,
            component_stats,
            mini_footprints,
            mini_denoised,
            subdir="iter/footprints/wrong",
            name="wrong",
        )

    @pytest.mark.viz
    def test_small_footprint(
        self, visualizer: Visualizer, mini_footprints: xr.DataArray, mini_denoised: xr.DataArray
    ) -> None:
        small_footprints = mini_footprints.copy()
        small_footprints[1] = binary_erosion(small_footprints[1])

        traces, pixel_stats, component_stats = self.get_stats(small_footprints, mini_denoised)

        self.visualize_iteration(
            visualizer,
            small_footprints,
            traces,
            pixel_stats,
            component_stats,
            mini_footprints,
            mini_denoised,
            subdir="iter/footprints/small",
            name="small",
        )

    @pytest.mark.viz
    def test_oversized_footprint(
        self, visualizer: Visualizer, mini_footprints: xr.DataArray, mini_denoised: xr.DataArray
    ) -> None:
        oversized_footprints = mini_footprints.copy()
        oversized_footprints[1] = binary_dilation(oversized_footprints[1])

        traces, pixel_stats, component_stats = self.get_stats(oversized_footprints, mini_denoised)

        self.visualize_iteration(
            visualizer,
            oversized_footprints,
            traces,
            pixel_stats,
            component_stats,
            mini_footprints,
            mini_denoised,
            subdir="iter/footprints/oversized",
            name="oversized",
        )

    @pytest.mark.viz
    def test_redundant_footprint(
        self, visualizer: Visualizer, mini_footprints: xr.DataArray, mini_denoised: xr.DataArray
    ) -> None:
        redundant_footprints = mini_footprints.copy()
        rolled = xr.DataArray(np.roll(mini_footprints[-1], -1), dims=("height", "width"))
        rolled = rolled.expand_dims("component").assign_coords(
            {"id_": ("component", ["id5"]), "type_": ("component", [Component.NEURON.value])}
        )
        redundant_footprints = xr.concat(
            [redundant_footprints, rolled],
            dim="component",
        )

        traces, pixel_stats, component_stats = self.get_stats(redundant_footprints, mini_denoised)

        self.visualize_iteration(
            visualizer,
            redundant_footprints,
            traces,
            pixel_stats,
            component_stats,
            mini_footprints,
            mini_denoised,
            subdir="iter/footprints/redundant",
            name="redundant",
        )
