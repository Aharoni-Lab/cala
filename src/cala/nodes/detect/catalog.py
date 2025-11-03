from collections.abc import Iterable
from itertools import compress
from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name
from noob.node import Node
from pydantic import Field
from scipy.ndimage import gaussian_filter1d
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from skimage.measure import label
from xarray import Coordinates

from cala.assets import Footprint, Footprints, Trace, Traces
from cala.models import AXIS
from cala.nodes.detect.slice_nmf import rank1nmf
from cala.util import combine_attr_replaces, create_id


class Cataloger(Node):
    smooth_kwargs: dict
    age_limit: int
    """Don't merge with new components if older than this number of frames."""
    merge_threshold: float
    val_threshold: float = Field(gt=0, lt=1)
    cnt_threshold: int = Field(gt=0)
    """must have cnt-number of pixels that are above the val-value"""

    def process(
        self,
        new_fps: list[Footprint],
        new_trs: list[Trace],
        existing_fp: Footprints | None = None,
        existing_tr: Traces | None = None,
    ) -> tuple[A[Footprints, Name("new_footprints")], A[Traces, Name("new_traces")]]:

        if not new_fps or not new_trs:
            return Footprints(), Traces()

        shape_chunks = xr.concat([fp.array for fp in new_fps], dim=AXIS.component_dim)
        trace_chunks = xr.concat([tr.array for tr in new_trs], dim=AXIS.component_dim)
        merge_mat = self._merge_matrix(shape_chunks, trace_chunks)
        new_shapes, new_traces = _merge(shape_chunks, trace_chunks, merge_mat)

        known_fp, known_tr = _get_absorption_targets(
            existing_fp.array, existing_tr.array, self.age_limit
        )
        merge_mat = self._merge_matrix(new_shapes, new_traces, known_fp, known_tr)
        footprints, traces = self._absorb(new_shapes, new_traces, known_fp, known_tr, merge_mat)
        # footprints = self._smooth(shapes)

        return Footprints.from_array(footprints), Traces.from_array(traces)

    def _smooth(self, shapes: xr.DataArray) -> xr.DataArray: ...

    def _merge_matrix(
        self,
        fps: xr.DataArray,
        trs: xr.DataArray,
        fps_base: csr_matrix | None = None,
        trs_base: xr.DataArray | None = None,
    ) -> xr.DataArray:
        fps = fps.stack(pixels=AXIS.spatial_dims)
        trs = xr.DataArray(
            gaussian_filter1d(trs.transpose(AXIS.component_dim, ...), **self.smooth_kwargs),
            dims=trs.dims,
            coords=trs.coords,
        )

        if fps_base is None:
            fps_base = fps.data
            trs_base = trs.rename({AXIS.component_dim: f"{AXIS.component_dim}'"})
        else:
            trs_base = xr.DataArray(
                gaussian_filter1d(
                    trs_base.transpose(AXIS.component_dim, ...), **self.smooth_kwargs
                ),
                dims=trs_base.dims,
                coords=trs_base.coords,
            )
            trs_base = trs_base.rename(AXIS.component_rename)

        overlaps = fps.data @ fps_base.T > 0
        # corr is fast. (~1ms to 4ms)
        corrs = xr.corr(trs, trs_base, dim=AXIS.frames_dim) > self.merge_threshold
        return xr.DataArray(overlaps * corrs.values, dims=corrs.dims, coords=corrs.coords)

    def _absorb(
        self,
        new_fps: xr.DataArray,
        new_trs: xr.DataArray,
        known_fps: csr_matrix,
        known_trs: xr.DataArray,
        merge_matrix: xr.DataArray,
    ) -> tuple[xr.DataArray | None, xr.DataArray | None]:
        footprints = []
        traces = []

        merge_matrix.data = label(merge_matrix.to_numpy(), background=0, connectivity=1)
        merge_matrix = merge_matrix.assign_coords(
            {AXIS.component_dim: range(merge_matrix.sizes[AXIS.component_dim])}
        ).reset_index(AXIS.component_dim)
        indep_idxs = (
            merge_matrix.where(merge_matrix.sum(f"{AXIS.component_dim}'") == 0, drop=True)[
                AXIS.component_dim
            ].values
            if known_fps is not None
            else np.array(range(len(merge_matrix)))
        )
        if indep_idxs.size > 0:
            fps, trs = _register_batch(
                new_fps=new_fps.isel({AXIS.component_dim: indep_idxs}),
                new_trs=new_trs.isel({AXIS.component_dim: indep_idxs}),
            )
            footprints.append(fps)
            traces.append(trs)

        num = merge_matrix.max().item()
        if num > 0 and known_fps is not None:
            for lbl in range(1, num + 1):
                new_idxs, _known_idxs = np.where(merge_matrix == lbl)
                fp = new_fps.sel({AXIS.component_dim: list(set(new_idxs))})
                tr = new_trs.sel({AXIS.component_dim: list(set(new_idxs))})
                footprint, trace = _merge_with(fp, tr, known_fps, known_trs, _known_idxs)

                footprints.append(footprint)
                traces.append(trace)

        mask = [np.sum(fp.data > self.val_threshold) > self.cnt_threshold for fp in footprints]
        footprints = list(compress(footprints, mask))
        traces = list(compress(traces, mask))

        if not footprints:
            return None, None

        footprints = xr.concat(
            footprints,
            dim=AXIS.component_dim,
            coords=[AXIS.id_coord, AXIS.detect_coord],
            combine_attrs=combine_attr_replaces,
        )
        traces = xr.concat(
            traces,
            dim=AXIS.component_dim,
            coords=[AXIS.id_coord, AXIS.detect_coord],
            combine_attrs=combine_attr_replaces,
        )

        return footprints, traces


def _get_absorption_targets(
    existing_fp: xr.DataArray, existing_tr: xr.DataArray, age_limit: int
) -> tuple[csr_matrix, xr.DataArray]:
    if existing_fp is not None:
        targets = existing_tr[AXIS.detect_coord].values > (
            existing_tr[AXIS.frame_coord].values.max() - age_limit
        )
        known_fp = existing_fp.data.reshape((existing_fp.sizes[AXIS.component_dim], -1)).tocsr()[
            targets
        ]
        known_tr = existing_tr[targets]
    else:
        known_fp = existing_fp
        known_tr = existing_tr
    return known_fp, known_tr


def _register(new_fp: xr.DataArray, new_tr: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:

    new_id = create_id()

    footprint = (
        new_fp.expand_dims(AXIS.component_dim)
        .assign_coords(
            {
                AXIS.id_coord: (AXIS.component_dim, [new_id]),
                AXIS.detect_coord: (
                    AXIS.component_dim,
                    [new_tr[AXIS.frame_coord].max().item()],
                ),
            }
        )
        .isel({AXIS.component_dim: 0})
    )
    trace = (
        new_tr.expand_dims(AXIS.component_dim)
        .assign_coords(
            {
                AXIS.id_coord: (AXIS.component_dim, [new_id]),
                AXIS.detect_coord: (
                    AXIS.component_dim,
                    [new_tr[AXIS.frame_coord].max().item()],
                ),
            }
        )
        .isel({AXIS.component_dim: 0})
    )

    return footprint, trace


def _register_batch(
    new_fps: xr.DataArray, new_trs: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    count = new_fps.sizes[AXIS.component_dim]
    new_ids = [create_id() for _ in range(count)]

    footprints = new_fps.assign_coords(
        {
            AXIS.id_coord: (AXIS.component_dim, new_ids),
            AXIS.detect_coord: (
                AXIS.component_dim,
                [new_trs[AXIS.frame_coord].max().item()] * count,
            ),
        }
    )
    traces = new_trs.assign_coords(
        {
            AXIS.id_coord: (AXIS.component_dim, new_ids),
            AXIS.detect_coord: (
                AXIS.component_dim,
                [new_trs[AXIS.frame_coord].max().item()] * count,
            ),
        }
    )

    return footprints, traces


def _recompose(
    movie: xr.DataArray, fp_coords: Coordinates, tr_coords: Coordinates
) -> tuple[xr.DataArray, xr.DataArray]:
    # Reshape neighborhood to 2D matrix (time × space)
    movie = movie.assign_coords({ax: movie[ax] for ax in AXIS.spatial_dims})
    shape = xr.DataArray(
        np.sum(movie.transpose(AXIS.frames_dim, ...).data, axis=0) > 0, dims=AXIS.spatial_dims
    )
    slice_ = movie.where(shape.as_numpy(), 0, drop=True)
    R = slice_.stack(space=AXIS.spatial_dims).transpose("space", AXIS.frames_dim)

    a, c, error = rank1nmf(R.values, np.mean(R.values, axis=1))

    a_new, c_new = _reshape(
        footprint=a,
        trace=c,
        fp_coords=fp_coords,
        tr_coords=tr_coords,
        slice_coords=slice_.coords,
    )

    factor = slice_.data.max() / c_new.data.max()
    a_new = a_new / factor
    c_new = c_new * factor

    return a_new, c_new


def _reshape(
    footprint: np.ndarray,
    trace: np.ndarray,
    fp_coords: Coordinates,
    tr_coords: Coordinates,
    slice_coords: Coordinates,
) -> tuple[xr.DataArray, xr.DataArray]:
    """Convert back to xarray with proper dimensions and coordinates"""

    c_new = xr.DataArray(trace.squeeze(), dims=[AXIS.frames_dim], coords=tr_coords)

    a_new = xr.DataArray(
        np.zeros(tuple(fp_coords.sizes.values())),
        dims=tuple(fp_coords.sizes.keys()),
        coords=fp_coords,
    )

    a_new.loc[slice_coords] = xr.DataArray(
        footprint.squeeze().reshape(list(slice_coords[ax].size for ax in AXIS.spatial_dims)),
        dims=AXIS.spatial_dims,
        coords=slice_coords,
    )

    return a_new, c_new


def _merge_with(
    new_fp: xr.DataArray,
    new_tr: xr.DataArray,
    target_fps: csr_matrix,
    target_trs: xr.DataArray,
    dupl_idx: Iterable[int],
) -> tuple[xr.DataArray, xr.DataArray]:
    target_fp = target_fps[dupl_idx]
    target_tr = target_trs[dupl_idx]

    recreated_movie = target_fp.T @ target_tr.dropna(dim=AXIS.frames_dim).data

    new_movie = np.matmul(
        new_fp.transpose(*AXIS.spatial_dims, ...).data,
        new_tr.dropna(dim=AXIS.frames_dim).data,
    )
    combined_movie = xr.DataArray(
        recreated_movie.reshape(new_movie.shape) + new_movie,
        dims=[*AXIS.spatial_dims, AXIS.frames_dim],
    )

    a_new, c_new = _recompose(
        combined_movie,
        new_fp.isel({AXIS.component_dim: 0}).coords,
        new_tr.isel({AXIS.component_dim: 0}).coords,
    )
    a_new.attrs["replaces"] = target_tr[AXIS.id_coord].values.tolist()
    c_new.attrs["replaces"] = target_tr[AXIS.id_coord].values.tolist()

    return _register(a_new, c_new)


def _merge(
    footprints: xr.DataArray, traces: xr.DataArray, merge_matrix: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray]:
    num, label = connected_components(merge_matrix.data)
    combined_fps = []
    combined_trs = []

    for lbl in set(label):
        group = np.where(label == lbl)[0]
        fps = footprints.sel({AXIS.component_dim: group})
        trs = traces.sel({AXIS.component_dim: group})
        if len(group) > 1:
            res = xr.DataArray(
                np.matmul(fps.transpose(*AXIS.spatial_dims, ...).data, trs.data),
                dims=[*AXIS.spatial_dims, AXIS.frames_dim],
            )
            new_fp, new_tr = _recompose(res, footprints[0].coords, traces[0].coords)
        else:
            new_fp, new_tr = fps[0], trs[0]
        combined_fps.append(new_fp)
        combined_trs.append(new_tr)

    new_fps = xr.concat(combined_fps, dim=AXIS.component_dim)
    new_trs = xr.concat(combined_trs, dim=AXIS.component_dim)

    return new_fps, new_trs
