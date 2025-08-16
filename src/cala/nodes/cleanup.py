from typing import Annotated as A

import cv2
import numpy as np
import xarray as xr
from noob import Name

from cala.assets import CompStats, Footprints, Overlaps, PixStats, Traces
from cala.models import AXIS


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
    keep_ids = _get_razed_ids(footprints=footprints, min_thicc=min_thicc)
    return filter_components(
        footprints=footprints,
        traces=traces,
        pix_stats=pix_stats,
        comp_stats=comp_stats,
        overlaps=overlaps,
        keep_ids=keep_ids,
    )


def _get_razed_ids(footprints: Footprints, min_thicc: int) -> A[xr.DataArray, Name("keep_ids")]:
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
    return A.isel({AXIS.component_dim: keep_idx})[AXIS.id_coord]


def filter_components(
    footprints: Footprints,
    traces: Traces,
    pix_stats: PixStats,
    comp_stats: CompStats,
    overlaps: Overlaps,
    keep_ids: xr.DataArray,
) -> tuple[
    A[Footprints, Name("footprints")],
    A[Traces, Name("traces")],
    A[PixStats, Name("pix_stats")],
    A[CompStats, Name("comp_stats")],
    A[Overlaps, Name("overlaps")],
]:
    if keep_ids.size == 0 or footprints.array is None:
        footprints.array = None
        traces.array = None
        pix_stats.array = None
        comp_stats.array = None
        overlaps.array = None

    elif not footprints.array[AXIS.id_coord].equals(keep_ids):
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
            .sel({AXIS.id_coord: keep_ids, f"{AXIS.id_coord}'": keep_ids.values.tolist()})
            .reset_index(AXIS.id_coord)
        )
        overlaps.array = (
            overlaps.array.set_xindex(AXIS.id_coord)
            .set_xindex(f"{AXIS.id_coord}'")
            .sel({AXIS.id_coord: keep_ids, f"{AXIS.id_coord}'": keep_ids.values.tolist()})
            .reset_index(AXIS.id_coord)
        )

    return footprints, traces, pix_stats, comp_stats, overlaps
