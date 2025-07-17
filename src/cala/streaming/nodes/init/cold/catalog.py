from collections.abc import Hashable, Mapping
from dataclasses import dataclass

import numpy as np
import xarray as xr
from sklearn.decomposition import NMF
from xarray import Coordinates

from cala.models.entity import Entities, Groups
from cala.models.params import Parameters
from cala.streaming.nodes import Node
from cala.streaming.util.new import create_id


@dataclass
class CatalogerParams(Parameters):

    def validate(self) -> bool: ...


@dataclass
class Cataloger(Node):
    params: CatalogerParams

    def process(
        self,
        new_fp: xr.DataArray,
        new_tr: xr.DataArray,
        existing_fp: xr.DataArray = None,
        existing_tr: xr.DataArray = None,
        duplicates: list[tuple[str, float]] | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        if not duplicates:
            footprints, traces = self._register(new_fp, new_tr, existing_fp, existing_tr)

        else:
            footprints, traces = self._merge(new_fp, new_tr, existing_fp, existing_tr, duplicates)

        return footprints, traces

    def _register(
        self,
        new_fp: xr.DataArray,
        new_tr: xr.DataArray,
        existing_fp: xr.DataArray | None = None,
        existing_tr: xr.DataArray | None = None,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        footprint, trace = self._init_with(new_fp, new_tr)

        if existing_fp is not None:
            footprint = xr.concat([existing_fp, footprint], dim=self.params.component_dim)
            trace = xr.concat([existing_tr, trace], dim=self.params.component_dim)

        return footprint, trace

    def _init_with(
        self, new_fp: xr.DataArray, new_tr: xr.DataArray, confidence: float = 0
    ) -> tuple[xr.DataArray, xr.DataArray]:
        new_fp.validate.against_schema(Entities.footprint.value)
        new_tr.validate.against_schema(Entities.trace.value)

        new_id = create_id()

        footprints = new_fp.expand_dims(self.params.component_dim).assign_coords(
            {
                self.params.id_coord: (self.params.component_dim, [new_id]),
                self.params.confidence_coord: (self.params.component_dim, [confidence]),
            }
        )
        traces = new_tr.expand_dims(self.params.component_dim).assign_coords(
            {
                self.params.id_coord: (self.params.component_dim, [new_id]),
                self.params.confidence_coord: (self.params.component_dim, [confidence]),
            }
        )

        return footprints, traces

    def _merge(
        self,
        new_fp: xr.DataArray,
        new_tr: xr.DataArray,
        existing_fp: xr.DataArray,
        existing_tr: xr.DataArray,
        duplicates: list[tuple[str, float]],
    ) -> tuple[xr.DataArray, xr.DataArray]:
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
        )

        # Reshape neighborhood to 2D matrix (time × space)
        shape = xr.DataArray(combined_movie.sum(dim=self.params.frames_dim) > 0).reset_coords(
            self.params.id_coord, drop=True
        )
        slice_ = combined_movie.where(shape, 0, drop=True)

        a, c = self._nmf(slice_)

        a_new, c_new = self._reshape(
            footprint=a,
            trace=c,
            frame_spec=shape.sizes,
            frame_coordinates=existing_tr[self.params.frames_dim].coords,
            spatial_coordinates=slice_.reset_index(self.params.frames_dim)
            .reset_coords(drop=True)
            .coords,
        )

        existing_tr.set_xindex(self.params.id_coord).loc[
            {self.params.id_coord: most_similar[0]}
        ] = c_new
        existing_fp.set_xindex(self.params.id_coord).loc[
            {self.params.id_coord: most_similar[0]}
        ] = a_new

        return existing_fp, existing_tr

    def _combine_component_movies(
        self,
        new_fp: xr.DataArray,
        new_tr: xr.DataArray,
        existing_fp: xr.DataArray,
        existing_tr: xr.DataArray,
        most_similar_id: str,
    ) -> xr.DataArray:
        most_similar_fp = existing_fp.set_xindex(self.params.id_coord).sel(
            {self.params.id_coord: most_similar_id}
        )
        most_similar_tr = existing_tr.set_xindex(self.params.id_coord).sel(
            {self.params.id_coord: most_similar_id}
        )

        most_similar_movie = most_similar_fp @ most_similar_tr
        new_movie = new_fp @ new_tr
        combined_movie = most_similar_movie + new_movie

        return combined_movie.assign_coords(
            {ax: combined_movie[ax] for ax in self.params.spatial_dims}
        )

    def _nmf(self, movie: xr.DataArray) -> tuple[np.ndarray, np.ndarray]:

        stacked = movie.stack({"space": self.params.spatial_dims}).transpose(
            self.params.frames_dim, "space"
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
        frame_spec: Mapping[Hashable, int],
        frame_coordinates: Coordinates,
        spatial_coordinates: Coordinates,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Convert back to xarray with proper dimensions and coordinates"""

        c_new = xr.DataArray(
            trace.squeeze(),
            dims=[self.params.frames_dim],
            coords=frame_coordinates,
        )

        a_new = xr.DataArray(np.zeros(tuple(frame_spec.values())), dims=tuple(frame_spec.keys()))

        a_new.loc[spatial_coordinates] = xr.DataArray(
            footprint.squeeze().reshape(
                list(spatial_coordinates[ax].size for ax in self.params.spatial_dims)
            ),
            dims=self.params.spatial_dims,
            coords=spatial_coordinates,
        )

        return a_new, c_new
