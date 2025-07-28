from collections.abc import Hashable, Mapping

import numpy as np
import xarray as xr
from noob.node import Node
from sklearn.decomposition import NMF

from cala.models import AXIS, Footprint, Residual, Trace


class SliceNMF(Node):
    cell_radius: int
    validity_threshold: float

    def process(self, residuals: Residual, energy: xr.DataArray) -> tuple[Footprint, Trace] | None:
        # Find and analyze neighborhood of maximum variance
        residuals = residuals.array

        slice_ = self._get_max_energy_slice(arr=residuals, energy_landscape=energy)

        a_new, c_new = self._local_nmf(
            slice_=slice_,
            spatial_sizes={k: v for k, v in residuals.sizes.items() if k in AXIS.spatial_dims},
        )

        # eventually we should just log this value instead of throwing out the component
        # otherwise we keep coming back to this energy max point
        if self._check_validity(a_new, residuals):
            return Footprint(array=a_new), Trace(array=c_new)
        else:
            return None

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

        # Apply NMF (check how long nndsvd takes vs random)
        model = NMF(n_components=1, init="nndsvd", tol=1e-4, max_iter=200)

        # when residual is negative, the error becomes massive...
        c = model.fit_transform(R.clip(0))  # temporal component
        a = model.components_  # spatial component

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

    def _check_validity(self, a_new: xr.DataArray, residuals: xr.DataArray) -> bool:
        # not sure if this step is necessary or even makes sense
        # how would a rank-1 nmf be not similar to the mean, unless the nmf error was massive?
        # and if the error is big, maybe it just means it's partially overlapping with another
        # luminescent object?
        # instead of tossing, we do candidates - cell, background, UNKNOWN
        # we gather everything. we merge everything as much as possible. and then we decide what to
        # do.

        # is it a cell / background, or just a plain wrong estimator - how do we distinguish these?
        # what i'm worried about is a WRONG estimator cannibalizing real cell signal
        # estimator_confidence, and neuron_confidence
        # estimator_confidence - how much of it can be explained by others
        # this can be remedied in the merge / split step? first it gets split, then it gets merged
        # neuron_confidence - how likely is it a neuron

        nonzero_ax_to_idx = {
            ax: sorted([int(x) for x in set(idx)])
            for ax, idx in zip(a_new.dims, a_new.values.nonzero())
        }
        # nonzero coordinates, like [[0, 1, 0, 1], [0, 0, 1, 1]]
        # for [0, 0], [0, 1], [1, 0], [1, 1]

        if len(list(nonzero_ax_to_idx.values())[0]) == 0:
            return False

        # it should look like something from a residual. paper does not specify this,
        # but i think we should only get correlation from within the new footprint perimeter,
        # since otherwise the correlation will get drowned out by the mismatch
        # from where the detected cell is NOT present.
        mean_residual = residuals.mean(dim=AXIS.frames_dim).isel(nonzero_ax_to_idx)

        a_norm = a_new.isel(nonzero_ax_to_idx) / a_new.sum()
        res_norm = mean_residual / mean_residual.sum()

        # the reason we break from detection as soon as this happens is because
        # we don't want to get flooded with wrong estimators.
        # we instead wait for when we get cleaner signal.
        # (this might not be super viable for cold-start.
        # what if the first cell is bad? we just keep trying?)
        r_spatial = xr.corr(a_norm, res_norm)

        return not r_spatial <= self.validity_threshold
