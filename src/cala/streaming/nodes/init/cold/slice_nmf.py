from dataclasses import dataclass
from typing import Hashable, Mapping

import numpy as np
import xarray as xr
from sklearn.decomposition import NMF
from xarray import Coordinates

from cala.streaming.core import Parameters, Axis
from cala.streaming.nodes import Node
from cala.streaming.stores.common import Footprints, Traces
from cala.streaming.stores.odl import Residuals


@dataclass
class SliceNMFParams(Parameters, Axis):
    cell_radius: int


@dataclass
class SliceNMF(Node):
    params: SliceNMFParams

    def process(self, residuals: Residuals, energy: xr.DataArray) -> tuple[Footprints, Traces]:
        # Find and analyze neighborhood of maximum variance
        neighborhood = self._get_max_energy_slice(arr=residuals, energy_landscape=energy)

        a_new, c_new = self._local_nmf(
            neighborhood=neighborhood,
            spatial_sizes=residuals.sizes,
            temporal_coords=residuals.coords,
        )

        return a_new, c_new

    def _get_max_energy_slice(
        self,
        arr: xr.DataArray,
        energy_landscape: xr.DataArray,
    ) -> xr.DataArray:
        """Find neighborhood around point of maximum variance."""
        # Find maximum point
        max_coords = energy_landscape.argmax(dim=self.params.spatial_axes)

        # Define neighborhood
        radius = int(np.round(self.params.cell_radius))
        window = {
            ax: slice(
                max(0, pos.values - radius),
                min(energy_landscape.sizes[ax], pos.values + radius + 1),
            )
            for ax, pos in max_coords.items()
        }

        # ok embed the actual coordinates onto the array
        neighborhood = arr.isel(window).assign_coords(
            {ax: energy_landscape.coords[ax][pos] for ax, pos in window.items()}
        )

        return neighborhood

    def _local_nmf(
        self,
        neighborhood: xr.DataArray,
        spatial_sizes: Mapping[Hashable, int],
        temporal_coords: Coordinates,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Perform local rank-1 Non-negative Matrix Factorization.

        Uses scikit-learn's NMF implementation to decompose the neighborhood
        into spatial (a) and temporal (c) components.

        Args:
            neighborhood (xr.DataArray): Local region of residual buffer.
                Shape: (frames × height × width)

        Returns:
            tuple[xr.DataArray, xr.DataArray]:
                - Spatial component a_new (height × width)
                - Temporal component c_new (frames)
        """
        # Reshape neighborhood to 2D matrix (time × space)
        R = neighborhood.stack(space=self.params.spatial_axes).transpose(
            self.params.frames_axis, "space"
        )

        # Apply NMF (check how long nndsvd takes vs random)
        model = NMF(n_components=1, init="nndsvd", tol=1e-4, max_iter=200)

        # when residual is negative, the error becomes massive...
        c = model.fit_transform(R.clip(0))  # temporal component
        a = model.components_  # spatial component

        # Convert back to xarray with proper dimensions and coordinates
        c_new = xr.DataArray(c.squeeze(), dims=[self.params.frames_axis], coords=temporal_coords)

        # Create full-frame zero array with proper coordinates
        a_new = xr.DataArray(
            np.zeros(tuple(spatial_sizes.values())), dims=tuple(spatial_sizes.keys())
        )

        # Place the NMF result in the correct location
        a_new.loc[{ax: neighborhood.coords[ax] for ax in self.params.spatial_axes}] = xr.DataArray(
            a.squeeze().reshape(tuple(neighborhood.sizes[ax] for ax in self.params.spatial_axes)),
            dims=self.params.spatial_axes,
            coords={ax: neighborhood.coords[ax] for ax in self.params.spatial_axes},
        )

        # normalize against the original video (as in whatever the residual used at the time)
        factor = neighborhood.max() / a_new.max()
        a_new = a_new * factor
        c_new = c_new / factor

        return a_new, c_new
