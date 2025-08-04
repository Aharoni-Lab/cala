from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name
from noob.node import Node
from sklearn.decomposition import NMF
from xarray import Coordinates

from cala.assets import Footprint, Footprints, Movie, Trace, Traces
from cala.models import AXIS
from cala.util import create_id


class Cataloger(Node):

    def process(
        self,
        new_fp: Footprint,
        new_tr: Trace,
        existing_fp: Footprints = None,
        existing_tr: Traces = None,
        duplicates: list[tuple[str, float]] | None = None,
    ) -> tuple[A[Footprint | None, Name("new_footprint")], A[Trace | None, Name("new_trace")]]:

        if not duplicates:
            footprint, trace = self._register(new_fp, new_tr)
        else:
            footprint, trace = self._merge(new_fp, new_tr, existing_fp, existing_tr, duplicates)

        return footprint, trace

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

    def _merge(
        self,
        new_fp: Footprint,
        new_tr: Trace,
        existing_fp: Footprints,
        existing_tr: Traces,
        duplicates: list[tuple[str, float]],
    ) -> tuple[Footprint, Trace]:
        """
        # 1. get the "movie" of the og piece
        # 2. get the "movie" of the new piece
        # 3. sum the movies
        # 4. do the NMF
        # 5. replace the original component with the merged one
        """

        most_similar = duplicates[0]
        most_similar_fp = existing_fp.array.set_xindex(AXIS.id_coord).sel(
            {AXIS.id_coord: most_similar[0]}
        )
        most_similar_tr = existing_tr.array.set_xindex(AXIS.id_coord).sel(
            {AXIS.id_coord: most_similar[0]}
        )

        combined_movie = self._combine_component_movies(
            new_fp=new_fp, new_tr=new_tr, fp_to_merge=most_similar_fp, tr_to_merge=most_similar_tr
        ).array

        # Reshape neighborhood to 2D matrix (time × space)
        shape = xr.DataArray(combined_movie.sum(dim=AXIS.frames_dim) > 0).reset_coords(
            [AXIS.id_coord, AXIS.confidence_coord], drop=True
        )
        slice_ = Movie.from_array(combined_movie.where(shape, 0, drop=True))

        a, c = self._nmf(slice_)

        slice_coords = slice_.array.reset_index(AXIS.frames_dim).reset_coords(drop=True).coords

        a_new, c_new = self._reshape(
            footprint=a,
            trace=c,
            fp_coords=most_similar_fp.coords,
            tr_coords=most_similar_tr.coords,
            slice_coords=slice_coords,
        )

        return a_new, c_new

    def _combine_component_movies(
        self, new_fp: Footprint, new_tr: Trace, fp_to_merge: xr.DataArray, tr_to_merge: xr.DataArray
    ) -> Movie:
        recreated_movie = fp_to_merge @ tr_to_merge
        new_movie = new_fp.array @ new_tr.array
        combined_movie = recreated_movie + new_movie

        return Movie.from_array(
            combined_movie.assign_coords({ax: combined_movie[ax] for ax in AXIS.spatial_dims})
        )

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
