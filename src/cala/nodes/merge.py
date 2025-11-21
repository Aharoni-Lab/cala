from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name
from scipy.ndimage import gaussian_filter1d
from scipy.sparse.csgraph import connected_components

from cala.assets import Footprints, Overlaps, Traces
from cala.models import AXIS
from cala.nodes.segment.catalog import _recompose, _register
from cala.util import combine_attr_replaces


def merge_existing(
    shapes: Footprints,
    traces: Traces,
    overlaps: Overlaps,
    merge_interval: int,
    merge_threshold: float,
    smooth_kwargs: dict,
    trigger: bool = None,
) -> tuple[A[Footprints, Name("footprints")], A[Traces, Name("traces")]]:
    """
    merge old ass components with each other, if they missed their
    natal window chance to get merged with their siamese twin
    """
    if overlaps.array is None:
        return Footprints(), Traces()

    idx = traces.array[AXIS.frame_coord].max().item()

    if idx % merge_interval != 0:
        return Footprints(), Traces()

    # only merge old components
    targets = traces.array[AXIS.detect_coord] <= (
        traces.array[AXIS.frame_coord].max() - merge_interval
    )

    if not any(targets):
        return Footprints(), Traces()

    target_ids = targets.where(targets, drop=True)[AXIS.id_coord].values

    target_fps, target_trs, target_ovs = _filter_targets(
        target_ids=target_ids,
        shapes=shapes,
        traces=traces,
        overlaps=overlaps,
        n_frames=merge_interval,
    )

    merge_mat = _merge_matrix(
        traces=target_trs,
        overlaps=target_ovs,
        smooth_kwargs=smooth_kwargs,
        threshold=merge_threshold,
    )

    num, label = connected_components(merge_mat.data)
    combined_fps = []
    combined_trs = []

    for lbl in set(label):
        group = np.where(label == lbl)[0]
        if len(group) <= 1:
            continue
        fps = target_fps.isel({AXIS.component_dim: group})
        trs = target_trs.isel({AXIS.component_dim: group})
        res = xr.DataArray(
            np.matmul(fps.transpose(*AXIS.spatial_dims, ...).data, trs.data),
            dims=[*AXIS.spatial_dims, AXIS.frames_dim],
        )
        a_new, c_new = _recompose(res, target_fps[0].coords, target_trs[0].coords)

        a_new.attrs["replaces"] = fps[AXIS.id_coord].values.tolist()
        c_new.attrs["replaces"] = trs[AXIS.id_coord].values.tolist()

        combined_fps.append(a_new)
        combined_trs.append(c_new)

    if len(combined_fps) == 0:
        return Footprints(), Traces()

    new_fps = xr.concat(
        combined_fps,
        dim=AXIS.component_dim,
        coords=[AXIS.id_coord, AXIS.detect_coord],
        combine_attrs=combine_attr_replaces,
    )
    new_trs = xr.concat(
        combined_trs,
        dim=AXIS.component_dim,
        coords=[AXIS.id_coord, AXIS.detect_coord],
        combine_attrs=combine_attr_replaces,
    )
    new_fps, new_trs = _register(new_fps, new_trs)

    return Footprints.from_array(new_fps), Traces.from_array(new_trs)


def _filter_targets(
    target_ids: np.ndarray, shapes: Footprints, traces: Traces, overlaps: Overlaps, n_frames: int
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """
    filter assets for old enough components
    """
    target_ovs = (
        overlaps.array.set_xindex([AXIS.id_coord])
        .set_xindex(f"{AXIS.id_coord}'")
        .sel({AXIS.id_coord: target_ids, f"{AXIS.id_coord}'": target_ids})
        .reset_index(AXIS.id_coord)
        .reset_index(f"{AXIS.id_coord}'")
    )

    target_trs = (
        traces.full_array(isel_filter={AXIS.frames_dim: slice(-n_frames, None)})
        .set_xindex(AXIS.id_coord)
        .sel({AXIS.id_coord: target_ids})
        .reset_index(AXIS.id_coord)
    ).transpose(AXIS.component_dim, ...)

    target_fps = (
        shapes.array.set_xindex(AXIS.id_coord)
        .sel({AXIS.id_coord: target_ids})
        .reset_index(AXIS.id_coord)
    )

    return target_fps, target_trs, target_ovs


def _merge_matrix(
    traces: xr.DataArray, overlaps: xr.DataArray, smooth_kwargs: dict, threshold: float
) -> xr.DataArray:
    """
    get merge matrix to use for connected_components
    """
    traces = xr.DataArray(
        gaussian_filter1d(traces.transpose(AXIS.component_dim, ...), **smooth_kwargs),
        dims=traces.dims,
        coords=traces.coords,
    )
    traces_base = traces.rename(AXIS.component_rename)

    corr = xr.corr(traces, traces_base, dim=AXIS.frames_dim)
    return overlaps * corr > threshold
