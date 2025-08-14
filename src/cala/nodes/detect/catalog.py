from typing import Annotated as A, Iterable, Hashable

import numpy as np
import xarray as xr
from noob import Name
from noob.node import Node
from scipy.sparse.csgraph import connected_components
from sklearn.decomposition import NMF
from xarray import Coordinates

from cala.assets import Footprint, Footprints, Movie, Trace, Traces
from cala.models import AXIS
from cala.util import create_id


class Cataloger(Node):
    merge_threshold: float

    def process(
        self,
        new_fps: list[Footprint],
        new_trs: list[Trace],
        existing_fp: Footprints | None = None,
        existing_tr: Traces | None = None,
    ) -> tuple[A[list[Footprint], Name("new_footprints")], A[list[Trace], Name("new_traces")]]:

        if not new_fps or not new_trs:
            return [], []

        existing_fp = existing_fp.array
        existing_tr = existing_tr.array

        new_fps = xr.concat([fp.array for fp in new_fps], dim=AXIS.component_dim)
        new_trs = xr.concat([tr.array for tr in new_trs], dim=AXIS.component_dim)

        conn_mat = self._connection_matrix(new_fps, new_trs)
        num, label = connected_components(conn_mat)
        combined_fps = []
        combined_trs = []

        for l in set(label):
            group = np.where(label == l)[0]
            fps = new_fps.sel({AXIS.component_dim: group})
            trs = new_trs.sel({AXIS.component_dim: group})
            res = fps @ trs
            new_fp, new_tr = self._decompose(res, new_fps[0].coords, new_trs[0].coords)

            combined_fps.append(new_fp)
            combined_trs.append(new_tr)

        new_fps = xr.concat([fp.array for fp in combined_fps], dim=AXIS.component_dim)
        new_trs = xr.concat([tr.array for tr in combined_trs], dim=AXIS.component_dim)

        conn_mat = self._connection_matrix(new_fps, new_trs, existing_fp, existing_tr)
        footprints = []
        traces = []

        # we're not doing connected components because it's not square matrix
        for i, dupes in enumerate(conn_mat.transpose(AXIS.component_dim, ...)):
            if not dupes or not any(dupes):
                footprint, trace = self._register(new_fps[i], new_trs[i])
            else:
                dupe_ids = dupes.where(dupes, drop=True)[f"{AXIS.id_coord}'"].values
                fp = new_fps.sel({AXIS.component_dim: i})
                tr = new_trs.sel({AXIS.component_dim: i})
                footprint, trace = self._merge_with(fp, tr, existing_fp, existing_tr, dupe_ids)

            footprints.append(footprint)
            traces.append(trace)

        return footprints, traces

    def _register(
        self, new_fp: Footprint, new_tr: Trace, confidence: float = 0.0
    ) -> tuple[Footprint, Trace]:

        new_id = create_id()

        footprint = (
            new_fp.array.expand_dims(AXIS.component_dim)
            .assign_coords(
                {
                    AXIS.id_coord: (AXIS.component_dim, [new_id]),
                    AXIS.confidence_coord: (AXIS.component_dim, [confidence]),
                }
            )
            .isel({AXIS.component_dim: 0})
        )
        trace = (
            new_tr.array.expand_dims(AXIS.component_dim)
            .assign_coords(
                {
                    AXIS.id_coord: (AXIS.component_dim, [new_id]),
                    AXIS.confidence_coord: (AXIS.component_dim, [confidence]),
                }
            )
            .isel({AXIS.component_dim: 0})
        )

        return Footprint.from_array(footprint), Trace.from_array(trace)

    def _merge_with(
        self,
        new_fp: xr.DataArray,
        new_tr: xr.DataArray,
        cognate_fps: xr.DataArray,
        cognate_trs: xr.DataArray,
        dupe_ids: Iterable[Hashable],
    ) -> tuple[Footprint, Trace]:
        cognate_fp = cognate_fps.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: dupe_ids})
        cognate_tr = cognate_trs.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: dupe_ids})

        recreated_movie = cognate_fp @ cognate_tr
        new_movie = new_fp @ new_tr
        combined_movie = recreated_movie + new_movie
        combined_movie = combined_movie.assign_coords(
            {ax: combined_movie[ax] for ax in AXIS.spatial_dims}
        )

        a_new, c_new = self._decompose(combined_movie, new_fp.coords, new_tr.coords)
        a_new.array.attrs["replaces"] = cognate_fp["id_"].values.tolist()
        c_new.array.attrs["replaces"] = cognate_tr["id_"].values.tolist()

        return self._register(a_new, c_new)

    def _decompose(
        self, movie: xr.DataArray, fp_coords: Coordinates, tr_coords: Coordinates
    ) -> tuple[Footprint, Trace]:
        # Reshape neighborhood to 2D matrix (time × space)
        shape = xr.DataArray(movie.sum(dim=AXIS.frames_dim) > 0)
        slice_ = Movie.from_array(movie.where(shape, 0, drop=True))

        a, c = self._nmf(slice_)

        slice_coords = slice_.array.reset_index(AXIS.frames_dim).reset_coords(drop=True).coords

        a_new, c_new = self._reshape(
            footprint=a,
            trace=c,
            fp_coords=fp_coords,
            tr_coords=tr_coords,
            slice_coords=slice_coords,
        )

        return a_new, c_new

    def _nmf(self, movie: Movie) -> tuple[np.ndarray, np.ndarray]:

        stacked = movie.array.stack({"space": AXIS.spatial_dims}).transpose(
            AXIS.frames_dim, "space"
        )
        # Apply NMF (check how long nndsvd takes vs random)
        model = NMF(n_components=1, init="nndsvd", tol=1e-4, max_iter=200)

        c = model.fit_transform(stacked)  # temporal component
        a = model.components_  # spatial component

        return a, c

    def _reshape(
        self,
        footprint: np.ndarray,
        trace: np.ndarray,
        fp_coords: Coordinates,
        tr_coords: Coordinates,
        slice_coords: Coordinates,
    ) -> tuple[Footprint, Trace]:
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

        return Footprint.from_array(a_new), Trace.from_array(c_new)

    def _connection_matrix(
        self,
        fps: xr.DataArray,
        trs: xr.DataArray,
        fps_base: xr.DataArray | None = None,
        trs_base: xr.DataArray | None = None,
    ) -> xr.DataArray:
        if fps_base is None:
            fps_base = fps.rename({AXIS.component_dim: f"{AXIS.component_dim}'"})
            trs_base = trs.rename({AXIS.component_dim: f"{AXIS.component_dim}'"})
        else:
            fps_base = fps_base.rename(AXIS.component_rename)
            trs_base = trs_base.rename(AXIS.component_rename)

        overlaps = fps @ fps_base > 0
        # this should later reflect confidence
        corrs = xr.corr(trs, trs_base, dim=AXIS.frames_dim) > self.merge_threshold

        return overlaps * corrs
