import numpy as np
import pytest
import xarray as xr

from cala.streaming.nodes.init.odl import ComponentStatsInitializer, PixelStatsInitializer
from cala.streaming.nodes.init.odl.component_stats import ComponentStatsInitializerParams
from cala.streaming.nodes.init.odl.overlaps import (
    OverlapsInitializer,
    OverlapsInitializerParams,
)
from cala.streaming.nodes.init.odl.pixel_stats import PixelStatsInitializerParams
from cala.streaming.nodes.iter.component_stats import (
    ComponentStatsUpdater,
    ComponentStatsUpdaterParams,
)
from cala.streaming.nodes.iter.detect import (
    Detector,
    DetectorParams,
)
from cala.streaming.nodes.iter.footprints import (
    FootprintsUpdater,
    FootprintsUpdaterParams,
)
from cala.streaming.nodes.iter.pixel_stats import (
    PixelStatsUpdater,
    PixelStatsUpdaterParams,
)
from cala.streaming.util import package_frame
from cala.viz_util import Visualizer


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

    @pytest.fixture
    def updater(self) -> Detector:
        return Detector(DetectorParams(num_nmf_residual_frames=5, gaussian_std=1))

    @pytest.fixture(scope="class")
    def overlapper(self) -> OverlapsInitializer:
        return OverlapsInitializer(OverlapsInitializerParams())

    @pytest.fixture(scope="class")
    def ps_initializer(self) -> PixelStatsInitializer:
        return PixelStatsInitializer(PixelStatsInitializerParams())

    @pytest.fixture(scope="class")
    def cs_initializer(self) -> ComponentStatsInitializer:
        return ComponentStatsInitializer(ComponentStatsInitializerParams())

    @pytest.mark.viz
    def test_missing_footprint(
        self,
        visualizer: Visualizer,
        updater: Detector,
        mini_footprints: xr.DataArray,
        mini_traces: xr.DataArray,
        mini_denoised: xr.DataArray,
        mini_pixel_stats: xr.DataArray,
        mini_component_stats: xr.DataArray,
        mini_overlaps: xr.DataArray,
    ) -> None:
        visualizer.plot_footprints(mini_footprints, subdir="iter/detect/footprints", name="label")
        visualizer.plot_traces(mini_traces, subdir="iter/detect/traces", name="label")
        visualizer.plot_trace_correlations(
            mini_traces, subdir="iter/detect/traces_corr", name="label"
        )
        # TODO: these two labels should be 4 minis and the 5th added according to the paper algo
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

        visualizer.plot_footprints(foot_missing, subdir="iter/detect/footprints", name="missing")
        visualizer.plot_traces(trace_missing, subdir="iter/detect/traces", name="missing")
        visualizer.plot_trace_correlations(
            trace_missing, subdir="iter/detect/traces_corr", name="missing"
        )
        visualizer.plot_pixel_stats(pixel_missing, subdir="iter/detect/pixel_stats", name="missing")
        visualizer.plot_component_stats(
            comp_missing, subdir="iter/detect/comp_stats", name="missing"
        )

        updater.learn_one(
            frame=package_frame(
                mini_denoised[-1].values,
                len(mini_denoised) - 1,
                mini_denoised[-1].coords["time_"].item(),
            ),
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

        fill_dims = dict(new_traces_.sizes)
        fill_dims["frame"] = trace_missing.sizes["frame"] - new_traces_.sizes["frame"]

        new_full_trs = xr.concat(
            [
                trace_missing,
                xr.concat(
                    [
                        xr.DataArray(
                            np.zeros(list(fill_dims.values())),
                            dims=fill_dims.keys(),
                            coords=trace_missing.isel(frame=slice(0, fill_dims["frame"]))
                            .coords["frame"]
                            .coords,
                        ),
                        new_traces_,
                    ],
                    dim="frame",
                ),
            ],
            dim="component",
        )

        visualizer.plot_footprints(new_full_fps, subdir="iter/detect/footprints", name="recovered")
        visualizer.plot_traces(new_full_trs, subdir="iter/detect/traces", name="recovered")
        visualizer.plot_trace_correlations(
            new_full_trs, subdir="iter/detect/traces_corr", name="recovered"
        )
        # TODO: Make sure the below two can recover the new footprint
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

        label_overlaps = mini_overlaps.copy()
        label_overlaps.values = label_overlaps.data.todense()
        overlap_missing.values = overlap_missing.data.todense()
        overlaps_.values = overlaps_.data.todense()

        visualizer.plot_overlaps(
            label_overlaps, mini_footprints, subdir="iter/detect/overlap", name="label"
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
        assert residuals_.max() < 1e-3

    @pytest.fixture
    def ps_updater(self) -> PixelStatsUpdater:
        return PixelStatsUpdater(PixelStatsUpdaterParams())

    @pytest.fixture
    def cs_updater(self) -> ComponentStatsUpdater:
        return ComponentStatsUpdater(ComponentStatsUpdaterParams())

    @pytest.fixture
    def fp_updater(self) -> FootprintsUpdater:
        return FootprintsUpdater(FootprintsUpdaterParams(boundary_expansion_pixels=1))

    @pytest.mark.viz
    def test_new_suff_stats(
        self,
        visualizer: Visualizer,
        updater: Detector,
        ps_initializer: PixelStatsUpdater,
        cs_initializer: ComponentStatsUpdater,
        ps_updater: PixelStatsInitializer,
        cs_updater: ComponentStatsInitializer,
        fp_updater: FootprintsUpdater,
        mini_footprints: xr.DataArray,
        mini_traces: xr.DataArray,
        mini_denoised: xr.DataArray,
        mini_overlaps: xr.DataArray,
    ) -> None:
        # 1. ps and cs were initialized before this incoming frame
        foot_missing = mini_footprints.isel(component=slice(None, -1))
        overlap_missing = mini_overlaps.isel(
            {"component": slice(None, -1), "component'": slice(None, -1)}
        )
        trace_missing = mini_traces.isel(component=slice(None, -1)).isel(
            frame=slice(None, -1)
        )  # before updating trace, so the last one hasn't come in yet
        pixel_missing = ps_initializer.learn_one(
            traces=trace_missing, frame=mini_denoised[:-1]
        ).transform_one()
        comp_missing = cs_initializer.learn_one(
            traces=trace_missing, frame=mini_denoised[:-1]
        ).transform_one()
        residual_missing = mini_denoised[:-1] - foot_missing @ trace_missing

        # 2. the new frame comes in
        incoming_frame = package_frame(
            mini_denoised[-1].values,
            len(mini_denoised) - 1,
            mini_denoised[-1].coords["time_"].item(),
        )

        # 3. traces were updated
        trace_updated = mini_traces.isel(
            component=slice(None, -1)
        )  # we've just updated the traces, so all frames

        pixel_stats_updated = ps_updater.learn_one(
            frame=incoming_frame, traces=trace_updated, pixel_stats=pixel_missing
        ).transform_one()
        component_stats_updated = cs_updater.learn_one(
            frame=incoming_frame, traces=trace_updated, component_stats=comp_missing
        ).transform_one()

        # 4. time to detect. we're going to get a component with the LATEST frame and attach it to everything.
        # so, everything should already be updated with the latest frame before we go in, as well!!
        updater.learn_one(
            frame=incoming_frame,
            footprints=foot_missing,  # footprints with a component missing. won't change just with a new frame.
            traces=trace_updated,
            # traces from a missing footprint (should try with both perfect & fucked up cause missing a footprint)
            residuals=residual_missing,  # residual (update happens inside detect. should probably come out.)
            overlaps=overlap_missing,  # overlaps won't change just with a new frame.
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
            traces=trace_updated,
            pixel_stats=pixel_stats_updated,
            component_stats=component_stats_updated,
            overlaps=overlap_missing,
        )

        new_full_fps = xr.concat([foot_missing, new_footprints_], dim="component")

        fill_dims = dict(new_traces_.sizes)
        fill_dims["frame"] = trace_updated.sizes["frame"] - new_traces_.sizes["frame"]

        # once updated, these should be 'identical' to mini_s. not sure how that will work without normalization.
        # confirmed, it's same except the newly found traces
        # think the order is wrong on the paper. suff-stats should be updated before detect for congruency.

        footprints_updated = fp_updater.learn_one(
            footprints=new_full_fps,
            pixel_stats=pixel_stats_,
            component_stats=component_stats_,
            frame=incoming_frame,  # same iteration
        ).transform_one()

        visualizer.plot_footprints(
            footprints_updated.transpose(*new_full_fps.dims),
            subdir="iter/detect/post_update",
        )

        assert True
