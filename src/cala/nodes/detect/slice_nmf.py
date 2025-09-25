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
from cala.logging import init_logger
from cala.models import AXIS


class SliceNMF(Node):
    min_frames: int
    """Wait until this number of frames to begin detecting."""
    detect_thresh: float
    """Minimum detection threshold for brightness fluctuation."""
    reprod_tol: float
    """Mean pixel value error tolerance for reproduced slice from the new component"""

    nmf_kwargs: dict[str, Any] = Field(default_factory=dict)

    error_: float = Field(None)
    _model: NMF = PrivateAttr(None)

    _logger = init_logger(__name__)

    def model_post_init(self, context: Any, /) -> None:
        self.nmf_kwargs.update({"n_components": 1, "init": "nndsvd"})
        self._model = NMF(**self.nmf_kwargs)

    def process(
        self, residuals: Residual, detect_radius: int
    ) -> tuple[A[list[Footprint], Name("new_fps")], A[list[Trace], Name("new_trs")]]:

        if residuals.array.sizes[AXIS.frames_dim] < self.min_frames:
            return [], []

        energy = self._get_energy(residuals.array)

        fps = []
        trs = []

        res = residuals.array.copy()

        while energy.max().item() >= self.detect_thresh:  # or use res directly
            # Find and analyze neighborhood of maximum variance
            slice_ = self._get_max_energy_slice(
                arr=res, energy_landscape=energy, radius=detect_radius
            )

            a_new, c_new = self._local_nmf(
                slice_=slice_,
                spatial_sizes={k: v for k, v in res.sizes.items() if k in AXIS.spatial_dims},
            )

            l1_norm = slice_.sum().item()
            comp_recon = a_new @ c_new

            energy.loc[{ax: slice_.coords[ax] for ax in AXIS.spatial_dims}] = 0

            if (self.error_ / l1_norm) <= self.reprod_tol:
                fps.append(Footprint.from_array(a_new))
                trs.append(Trace.from_array(c_new))
                res = (res - comp_recon).clip(0)
            else:
                l0_norm = np.prod(slice_.shape)
                res.loc[{ax: slice_.coords[ax] for ax in AXIS.spatial_dims}] = self.error_ / l0_norm

        return fps, trs

    def _get_energy(self, res: xr.DataArray) -> xr.DataArray:
        # should technically be median but it's so slow
        pixels_median = res.mean(dim=AXIS.frames_dim)
        V = res - pixels_median

        return np.sqrt((V**2).mean(dim=AXIS.frames_dim))

    def _get_max_energy_slice(
        self,
        arr: xr.DataArray,
        energy_landscape: xr.DataArray,
        radius: int,
    ) -> xr.DataArray:
        """Find neighborhood around point of maximum variance."""
        # Find maximum point
        max_coords = energy_landscape.argmax(dim=AXIS.spatial_dims)

        # Define neighborhood
        window = {
            ax: slice(
                max(0, pos.values - radius),
                min(energy_landscape.sizes[ax], pos.values + radius + 1),
            )
            for ax, pos in max_coords.items()
        }

        # Embed the actual coordinates onto the array
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

        self.error_ = self._model.reconstruction_err_.item()

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
