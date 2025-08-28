from typing import Annotated as A

import cv2
import numpy as np
import xarray as xr
from noob import Name

from cala.assets import CompStats, Footprints, Overlaps, PixStats, Residual, Traces, Frame, PopSnap
from cala.models import AXIS


def clear_overestimates(
    footprints: Footprints, residuals: Residual, nmf_error: float
) -> A[Footprints, Name("footprints")]:
    """
    Remove all sections of the footprints that cause negative residuals.

    This occurs by:
    1. find "significant" negative residual spots that is more than a noise level, and thus
    cannot be clipped to zero. !!!! (only of the latest frame, and then go back to trace update..?)
    2. all footprint values at these spots go to zero.
    """
    if residuals.array is None:
        return footprints
    R_min = residuals.array.isel({AXIS.frames_dim: -1}).reset_coords(
        [AXIS.frame_coord, AXIS.timestamp_coord], drop=True
    )
    tuned_fp = footprints.array.where(R_min > -nmf_error, 0, drop=False)

    return tuned_fp


def purge_razed_components(
    footprints: Footprints,
    traces: Traces,
    pix_stats: PixStats,
    comp_stats: CompStats,
    overlaps: Overlaps,
    min_thicc: int,
    trigger: bool,
) -> tuple[
    A[Footprints, Name("footprints")],
    A[Traces, Name("traces")],
    A[PixStats, Name("pix_stats")],
    A[CompStats, Name("comp_stats")],
    A[Overlaps, Name("overlaps")],
]:
    keep_ids = _filter_razed_ids(footprints=footprints, min_thicc=min_thicc)
    return _filter_components(
        footprints=footprints,
        traces=traces,
        pix_stats=pix_stats,
        comp_stats=comp_stats,
        overlaps=overlaps,
        keep_ids=keep_ids,
    )


def _filter_razed_ids(footprints: Footprints, min_thicc: int) -> A[list[str], Name("keep_ids")]:
    """
    :param min_thicc: minimum number of pixel thickness to keep the cell
    :return:
    """
    A = footprints.array

    if A is None:
        return xr.DataArray([])

    kernel = np.ones((min_thicc, min_thicc), np.uint8)

    eroded = xr.apply_ufunc(
        cv2.erode,
        (A > 0).astype(np.uint8),
        kwargs={"kernel": kernel},
        vectorize=True,
        input_core_dims=[AXIS.spatial_dims],
        output_core_dims=[AXIS.spatial_dims],
    )

    keep_idx = np.where(eroded.sum(dim=AXIS.spatial_dims).values.tolist())[0]
    return A.isel({AXIS.component_dim: keep_idx})[AXIS.id_coord].values.tolist()


def _filter_components(
    footprints: Footprints,
    traces: Traces,
    pix_stats: PixStats,
    comp_stats: CompStats,
    overlaps: Overlaps,
    keep_ids: list[str],
) -> tuple[
    A[Footprints, Name("footprints")],
    A[Traces, Name("traces")],
    A[PixStats, Name("pix_stats")],
    A[CompStats, Name("comp_stats")],
    A[Overlaps, Name("overlaps")],
]:
    if len(keep_ids) == 0 or footprints.array is None:
        footprints.array = None
        traces.array = None
        pix_stats.array = None
        comp_stats.array = None
        overlaps.array = None

    elif not footprints.array[AXIS.id_coord].values.tolist() == keep_ids:
        footprints.array = (
            footprints.array.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )
        traces.array = (
            traces.array.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )
        pix_stats.array = (
            pix_stats.array.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )
        comp_stats.array = (
            comp_stats.array.set_xindex(AXIS.id_coord)
            .set_xindex(f"{AXIS.id_coord}'")
            .sel({AXIS.id_coord: keep_ids, f"{AXIS.id_coord}'": keep_ids})
            .reset_index([AXIS.id_coord, f"{AXIS.id_coord}'"])
        )
        overlaps.array = (
            overlaps.array.set_xindex(AXIS.id_coord)
            .set_xindex(f"{AXIS.id_coord}'")
            .sel({AXIS.id_coord: keep_ids, f"{AXIS.id_coord}'": keep_ids})
            .reset_index([AXIS.id_coord, f"{AXIS.id_coord}'"])
        )

    return footprints, traces, pix_stats, comp_stats, overlaps


def _filter_redundant(
    footprints: Footprints,
    traces: Traces,
    frame: Frame,
    min_life_in_frames: int,
    quantile: float = 0.8,
    rel_threshold: float = 0.9,
    abs_threshold: float = 1.0,
) -> list[str]:
    """
    Remove redundant components
    Tested with SplitOffSource

    1. should have been some time since discovery
    2. max of residual over the last however many frames is very similar to y / trending up % wise

    :param quantile: the higher, the more stringent
    """
    A = footprints.array
    c_t = traces.array.isel({AXIS.frames_dim: -1})
    y_t = frame.array

    keep_ids = []
    for a, c in zip(A.transpose(AXIS.component_dim, ...), c_t.transpose(AXIS.component_dim, ...)):
        if y_t[AXIS.frame_coord] - a[AXIS.detect_coord] < min_life_in_frames:
            keep_ids.append(a[AXIS.id_coord].item())

        ratio = (a @ c / y_t).where(a, np.nan, drop=True).quantile(1 - quantile)
        diff = np.abs((a @ c - y_t).where(a, np.nan, drop=True)).quantile(quantile)

        if ratio > rel_threshold and diff < abs_threshold:
            keep_ids.append(a[AXIS.id_coord].item())

    return keep_ids


def merge_components(
    footprints: Footprints,
    traces: Traces,
) -> A[Footprints, Name("footprints")]:
    """
    Merge existing components
    """
