import numpy as np
import pytest
import xarray as xr

from cala.streaming.composer import Frame
from cala.streaming.init.odl import PixelStatsInitializer, ComponentStatsInitializer
from cala.streaming.init.odl.component_stats import ComponentStatsInitializerParams
from cala.streaming.init.odl.overlaps import (
    OverlapsInitializer,
    OverlapsInitializerParams,
)
from cala.streaming.init.odl.pixel_stats import PixelStatsInitializerParams
from cala.streaming.iterate.detect import (
    Detector,
    DetectorParams,
)


class TestDetector:
    """need to inject:
    frame: Frame,
    footprints: Footprints,
    traces: Traces,
    residuals: Residuals,
    overlaps: Overlaps,

    verify after:
    footprints: Footprints,
    traces: Traces,
    pixel_stats: PixelStats,
    component_stats: ComponentStats,
    overlaps: Overlaps,
    """

    @pytest.fixture(scope="class")
    def updater(self):
        return Detector(DetectorParams(num_nmf_residual_frames=3, gaussian_radius=1))

    @pytest.fixture(scope="class")
    def overlapper(self):
        return OverlapsInitializer(OverlapsInitializerParams())

    @pytest.fixture(scope="class")
    def ps_initializer(self):
        return PixelStatsInitializer(PixelStatsInitializerParams())

    @pytest.fixture(scope="class")
    def cs_initializer(self):
        return ComponentStatsInitializer(ComponentStatsInitializerParams())

    @pytest.mark.viz
    def test_missing_footprint(
        self,
        visualizer,
        updater,
        mini_footprints,
        mini_traces,
        mini_denoised,
        mini_pixel_stats,
        mini_component_stats,
        mini_overlaps,
    ):
        visualizer.plot_footprints(
            mini_footprints, subdir="iter/detect/footprints", name="label"
        )
        visualizer.plot_traces(mini_traces, subdir="iter/detect/traces", name="label")
        visualizer.plot_trace_correlations(
            mini_traces, subdir="iter/detect/traces_corr", name="label"
        )
        visualizer.plot_pixel_stats(
            mini_pixel_stats, subdir="iter/detect/pixel_stats", name="label"
        )
        visualizer.plot_component_stats(
            mini_component_stats, subdir="iter/detect/comp_stats", name="label"
        )

        foot_missing = mini_footprints.isel(component=slice(None, -1))
        trace_missing = mini_traces.isel(
            component=slice(None, -1)
        )  # we've just updated the traces, so all frames
        residual_missing = (mini_denoised - foot_missing @ trace_missing).isel(
            frame=slice(None, -1)  # residual not yet (gets updated during detect)
        )
        pixel_missing = mini_pixel_stats.isel(component=slice(None, -1))
        comp_missing = mini_component_stats.isel(
            {"component": slice(None, -1), "component'": slice(None, -1)}
        )
        overlap_missing = mini_overlaps.isel(
            {"component": slice(None, -1), "component'": slice(None, -1)}
        )

        visualizer.plot_footprints(
            foot_missing, subdir="iter/detect/footprints", name="missing"
        )
        visualizer.plot_traces(
            trace_missing, subdir="iter/detect/traces", name="missing"
        )
        visualizer.plot_trace_correlations(
            trace_missing, subdir="iter/detect/traces_corr", name="missing"
        )
        visualizer.plot_pixel_stats(
            pixel_missing, subdir="iter/detect/pixel_stats", name="missing"
        )
        visualizer.plot_component_stats(
            comp_missing, subdir="iter/detect/comp_stats", name="missing"
        )

        updater.learn_one(
            frame=Frame(mini_denoised[-1], len(mini_denoised)),
            footprints=foot_missing,  # footprints with a component missing
            traces=trace_missing,
            # traces from a missing footprint (should try with both perfect & fucked up cause missing a footprint)
            residuals=residual_missing,  # residual (same as traces)
            overlaps=overlap_missing,  # overlaps won't change
        )

        (
            new_footprints_,
            new_traces_,
            residuals_,
            pixel_stats_,
            component_stats_,
            overlaps_,
        ) = updater.transform_one(
            footprints=foot_missing,
            traces=trace_missing,
            pixel_stats=pixel_missing,
            component_stats=comp_missing,
            overlaps=overlap_missing,
        )

        new_fp_l2 = new_footprints_.sum(dim=("height", "width"))
        label_l2 = mini_footprints[-1].sum(dim=("height", "width"))
        trace_len = new_traces_.sizes["frame"]

        new_full_fps = xr.concat([foot_missing, new_footprints_], dim="component")
        new_full_trs = xr.concat(
            [
                trace_missing,
                xr.concat(
                    [xr.DataArray([[0, 0]], dims=new_traces_.dims), new_traces_],
                    dim="frame",
                ),
            ],
            dim="component",
        )

        visualizer.plot_footprints(
            new_full_fps, subdir="iter/detect/footprints", name="recovered"
        )
        visualizer.plot_traces(
            new_full_trs, subdir="iter/detect/traces", name="recovered"
        )
        visualizer.plot_trace_correlations(
            new_full_trs, subdir="iter/detect/traces_corr", name="recovered"
        )
        visualizer.plot_pixel_stats(
            pixel_stats_, subdir="iter/detect/pixel_stats", name="recovered"
        )
        visualizer.plot_component_stats(
            component_stats_, subdir="iter/detect/comp_stats", name="recovered"
        )

        mini_recovery = (new_full_fps @ new_full_trs).transpose(*mini_denoised.dims)

        visualizer.save_video_frames(
            [
                (mini_denoised, "label"),
                (
                    (foot_missing @ trace_missing).transpose(*mini_denoised.dims),
                    "missing",
                ),
                (mini_recovery, "recovered"),
            ],
            subdir="iter/detect",
        )

        mini_overlaps.values = mini_overlaps.data.todense()
        overlap_missing.values = overlap_missing.data.todense()
        overlaps_.values = overlaps_.data.todense()

        visualizer.plot_overlaps(
            mini_overlaps, mini_footprints, subdir="iter/detect/overlap", name="label"
        )
        visualizer.plot_overlaps(
            overlap_missing, foot_missing, subdir="iter/detect/overlap", name="missing"
        )
        visualizer.plot_overlaps(
            overlaps_, new_full_fps, subdir="iter/detect/overlap", name="recovered"
        )

        assert np.allclose(new_footprints_ / new_fp_l2, mini_footprints[-1] / label_l2)
        assert np.allclose(
            new_traces_ * new_fp_l2,
            (mini_traces[-1] * label_l2).isel(frame=slice(-trace_len, None)),
        )
