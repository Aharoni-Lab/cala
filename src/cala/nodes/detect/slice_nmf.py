from collections.abc import Hashable, Mapping
from typing import Annotated as A
from typing import Any

import numpy as np
import xarray as xr
from noob import Name
from noob.node import Node
from pydantic import Field, PrivateAttr
from sklearn.decomposition import NMF

from cala.assets import Footprint, Residual, Trace
from cala.models import AXIS


class SliceNMF(Node):
    cell_radius: int
    nmf_kwargs: dict[str, Any] = Field(default_factory=dict)
    validity_threshold: float

    errors_: list[float] = Field(default_factory=list)
    _model: NMF = PrivateAttr(None)

    def model_post_init(self, context: Any, /) -> None:
        self.nmf_kwargs.update({"n_components": 1, "init": "nndsvd"})
        if not self.nmf_kwargs.get("tol", None):
            self.nmf_kwargs["tol"] = 1e-4

        self._model = NMF(**self.nmf_kwargs)

    def process(
        self, residuals: Residual, energy: xr.DataArray
    ) -> tuple[A[Footprint, Name("new_fp")], A[Trace, Name("new_tr")]]:
        residuals = residuals.array

        if energy.size > 1 and residuals.max().item() > self.nmf_kwargs["tol"]:
            # Find and analyze neighborhood of maximum variance
            slice_ = self._get_max_energy_slice(arr=residuals, energy_landscape=energy)

            a_new, c_new = self._local_nmf(
                slice_=slice_,
                spatial_sizes={k: v for k, v in residuals.sizes.items() if k in AXIS.spatial_dims},
            )

            # eventually we should just log this value instead of throwing out the component
            # otherwise we keep coming back to this energy max point
            if (self.errors_[-1] / slice_.sum().item()) <= self.nmf_kwargs["tol"]:
                return Footprint.from_array(a_new), Trace.from_array(c_new)
        return Footprint(), Trace()

    def _get_max_energy_slice(
        self,
        arr: xr.DataArray,
        energy_landscape: xr.DataArray,
    ) -> xr.DataArray:
        """Find neighborhood around point of maximum variance."""
        # Find maximum point
        max_coords = energy_landscape.argmax(dim=AXIS.spatial_dims)

        # Define neighborhood
        radius = int(np.round(self.cell_radius))
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
        slice_: xr.DataArray,
        spatial_sizes: Mapping[Hashable, int],
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Perform local rank-1 Non-negative Matrix Factorization.

        Uses scikit-learn's NMF implementation to decompose the neighborhood
        into spatial (a) and temporal (c) components.

        Args:
            slice_ (xr.DataArray): Local region of residual buffer.
                Shape: (frames × height × width)

        Returns:
            tuple[xr.DataArray, xr.DataArray]:
                - Spatial component a_new (height × width)
                - Temporal component c_new (frames)
        """
        # Reshape neighborhood to 2D matrix (time × space)
        R = slice_.stack(space=AXIS.spatial_dims).transpose(AXIS.frames_dim, "space")

        c = self._model.fit_transform(R)  # temporal component
        a = self._model.components_  # spatial component

        err = self._model.reconstruction_err_.item()
        self.errors_.append(err)

        # Convert back to xarray with proper dimensions and coordinates
        c_new = xr.DataArray(
            c.squeeze(),
            dims=[AXIS.frames_dim],
            coords=slice_[AXIS.frames_dim].coords,
        )

        # Create full-frame zero array with proper coordinates
        a_new = xr.DataArray(
            np.zeros(tuple(spatial_sizes.values())),
            dims=tuple(spatial_sizes.keys()),
            coords={ax: np.arange(size) for ax, size in spatial_sizes.items()},
        )

        # Place the NMF result in the correct location
        a_new.loc[{ax: slice_.coords[ax] for ax in AXIS.spatial_dims}] = xr.DataArray(
            a.squeeze().reshape(tuple(slice_.sizes[ax] for ax in AXIS.spatial_dims)),
            dims=AXIS.spatial_dims,
            coords={ax: slice_.coords[ax] for ax in AXIS.spatial_dims},
        )

        # normalize against the original video (as in whatever the residual used at the time)
        factor = slice_.max() / a_new.max()
        a_new = a_new * factor
        c_new = c_new / factor

        return a_new, c_new
