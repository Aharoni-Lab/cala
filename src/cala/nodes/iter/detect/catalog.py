import numpy as np
import xarray as xr
from noob.node import Node
from sklearn.decomposition import NMF
from xarray import Coordinates

from cala.assets import Footprint, Footprints, Movie, Trace, Traces
from cala.models import AXIS
from cala.util.new import create_id


class Cataloger(Node):

    def process(
        self,
        new_fp: Footprint,
        new_tr: Trace,
        existing_fp: Footprints = None,
        existing_tr: Traces = None,
        duplicates: list[tuple[str, float]] | None = None,
    ) -> tuple[Footprints, Traces]:

        if not duplicates:
            footprints, traces = self._register(new_fp, new_tr, existing_fp, existing_tr)

        else:
            footprints, traces = self._merge(new_fp, new_tr, existing_fp, existing_tr, duplicates)

        return footprints, traces

    def _register(
        self,
        new_fp: Footprint,
        new_tr: Trace,
        existing_fp: Footprints | None = None,
        existing_tr: Traces | None = None,
    ) -> tuple[Footprints, Traces]:
        footprint, trace = self._init_with(new_fp, new_tr)

        if existing_fp is not None:
            footprint = xr.concat([existing_fp.array, footprint.array], dim=AXIS.component_dim)
            trace = xr.concat([existing_tr.array, trace.array], dim=AXIS.component_dim)

        return Footprints(array=footprint), Traces(array=trace)

    def _init_with(
        self, new_fp: Footprint, new_tr: Trace, confidence: float = 0.0
    ) -> tuple[Footprints, Traces]:

        new_id = create_id()

        footprints = new_fp.array.expand_dims(AXIS.component_dim).assign_coords(
            {
                AXIS.id_coord: (AXIS.component_dim, [new_id]),
                AXIS.confidence_coord: (AXIS.component_dim, [confidence]),
            }
        )
        traces = new_tr.array.expand_dims(AXIS.component_dim).assign_coords(
            {
                AXIS.id_coord: (AXIS.component_dim, [new_id]),
                AXIS.confidence_coord: (AXIS.component_dim, [confidence]),
            }
        )

        return Footprints(array=footprints), Traces(array=traces)

    def _merge(
        self,
        new_fp: Footprint,
        new_tr: Trace,
        existing_fp: Footprints,
        existing_tr: Traces,
        duplicates: list[tuple[str, float]],
    ) -> tuple[Footprints, Traces]:
        """
        # 1. get the "movie" of the og piece
        # 2. get the "movie" of the new piece
        # 3. sum the movies
        # 4. do the NMF
        # 5. replace the original component with the merged one
        """

        most_similar = duplicates[0]

        combined_movie = self._combine_component_movies(
            new_fp, new_tr, existing_fp, existing_tr, most_similar[0]
        ).array

        # Reshape neighborhood to 2D matrix (time × space)
        shape = xr.DataArray(combined_movie.sum(dim=AXIS.frames_dim) > 0).reset_coords(
            [AXIS.id_coord, AXIS.confidence_coord], drop=True
        )
        slice_ = Movie(array=combined_movie.where(shape, 0, drop=True))

        a, c = self._nmf(slice_)

        a_new, c_new = self._reshape(
            footprint=a,
            trace=c,
            template_coords=shape.coords,
            frame_coords=existing_tr.array[AXIS.frames_dim].coords,
            slice_coords=slice_.array.reset_index(AXIS.frames_dim).reset_coords(drop=True).coords,
        )

        footprints, traces = existing_fp.array.copy(), existing_tr.array.copy()

        traces.set_xindex(AXIS.id_coord).loc[{AXIS.id_coord: most_similar[0]}] = c_new.array

        footprints.set_xindex(AXIS.id_coord).loc[{AXIS.id_coord: most_similar[0]}] = a_new.array

        return Footprints(array=footprints), Traces(array=traces)

    def _combine_component_movies(
        self,
        new_fp: Footprint,
        new_tr: Trace,
        existing_fp: Footprints,
        existing_tr: Traces,
        most_similar_id: str,
    ) -> Movie:
        most_similar_fp = existing_fp.array.set_xindex(AXIS.id_coord).sel(
            {AXIS.id_coord: most_similar_id}
        )
        most_similar_tr = existing_tr.array.set_xindex(AXIS.id_coord).sel(
            {AXIS.id_coord: most_similar_id}
        )

        most_similar_movie = most_similar_fp @ most_similar_tr
        new_movie = new_fp.array @ new_tr.array
        combined_movie = most_similar_movie + new_movie

        return Movie(
            array=combined_movie.assign_coords({ax: combined_movie[ax] for ax in AXIS.spatial_dims})
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
        template_coords: Coordinates,
        frame_coords: Coordinates,
        slice_coords: Coordinates,
    ) -> tuple[Footprint, Trace]:
        """Convert back to xarray with proper dimensions and coordinates"""

        c_new = xr.DataArray(
            trace.squeeze(),
            dims=[AXIS.frames_dim],
            coords=frame_coords,
        )

        a_new = xr.DataArray(
            np.zeros(tuple(template_coords.sizes.values())),
            dims=tuple(template_coords.sizes.keys()),
            coords=template_coords,
        )

        a_new.loc[slice_coords] = xr.DataArray(
            footprint.squeeze().reshape(list(slice_coords[ax].size for ax in AXIS.spatial_dims)),
            dims=AXIS.spatial_dims,
            coords=slice_coords,
        )

        return Footprint(array=a_new), Trace(array=c_new)
