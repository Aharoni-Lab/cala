from dataclasses import dataclass
from typing import Self

import cv2
import numpy as np
import xarray as xr
from river.base import SupervisedTransformer
from sklearn.exceptions import NotFittedError

from cala.streaming.core import Axis, Parameters
from cala.streaming.stores.common import Footprints
from cala.streaming.stores.odl import ComponentStats, PixelStats


@dataclass
class FootprintsUpdaterParams(Parameters, Axis):
    """Parameters for spatial footprint updates.

    This class defines the configuration parameters needed for updating
    spatial footprints of components, including axis names and iteration limits.
    """

    boundary_expansion_pixels: int | None = None
    """Number of pixels to explore the boundary of the footprint outside of the current footprint."""

    tolerance: float = 1e-7

    def validate(self) -> None:
        """Validate parameter configurations.

        Raises:
            ValueError: If max_iterations is not positive.
        """
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


@dataclass
class FootprintsUpdater(SupervisedTransformer):
    """Updates spatial footprints using sufficient statistics.

    This transformer implements Algorithm 6 (UpdateShapes) which updates
    the spatial footprints of components using pixel-wise and component-wise
    sufficient statistics. The update follows the equation:

    Ã[p, i] = max(Ã[p, i] + (W[p, i] - Ã[p, :]M[i, :])/M[i, i], 0)

    where:
    - Ã is the spatial footprints matrix
    - W is the pixel-wise sufficient statistics
    - M is the component-wise sufficient statistics
    - p are the pixels where component i can be non-zero
    """

    params: FootprintsUpdaterParams
    """Configuration parameters for the update process."""

    footprints_: xr.DataArray = None
    """Updated spatial footprints matrix."""

    is_fitted_: bool = False
    """Indicator whether the transformer has been fitted."""

    def learn_one(
        self,
        footprints: Footprints,
        pixel_stats: PixelStats,
        component_stats: ComponentStats,
        frame: xr.DataArray = None,
    ) -> Self:
        """Update spatial footprints using sufficient statistics.

        This method implements the iterative update of spatial footprints
        for specified components. The update process maintains non-negativity
        constraints while optimizing the footprint shapes based on accumulated
        statistics.

        Args:
            footprints (Footprints): Current spatial footprints Ã = [A, b].
                Shape: (pixels × components)
            pixel_stats (PixelStats): Sufficient statistics W.
                Shape: (pixels × components)
            component_stats (ComponentStats): Sufficient statistics M.
                Shape: (components × components)
            frame (Frame): Streaming frame (Unused).

        Returns:
            Self: The transformer instance for method chaining.
        """
        A = footprints
        M = component_stats
        side_length = min(
            footprints.sizes[self.params.spatial_axes[0]],
            footprints.sizes[self.params.spatial_axes[1]],
        )
        if self.params.boundary_expansion_pixels:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_CROSS,
                (
                    self.params.boundary_expansion_pixels * 2 + 1,
                    self.params.boundary_expansion_pixels * 2 + 1,
                ),
            )  # faster than np.ones

        converged = False
        count = 0
        while not converged:
            count += 1
            mask = A > 0
            if self.params.boundary_expansion_pixels and count < side_length:
                mask = xr.apply_ufunc(
                    lambda x: cv2.morphologyEx(x, cv2.MORPH_DILATE, kernel, iterations=1),
                    mask.astype(np.uint8),
                    input_core_dims=[[*self.params.spatial_axes]],
                    output_core_dims=[[*self.params.spatial_axes]],
                    vectorize=True,
                    dask="parallelized",
                )
            # Compute AM product using xarray operations
            # Reshape M to align dimensions for broadcasting
            AM = (A @ M).rename({f"{self.params.component_axis}'": f"{self.params.component_axis}"})
            numerator = pixel_stats - AM

            # Compute update using vectorized operations
            # Expand M diagonal for broadcasting
            M_diag = xr.apply_ufunc(
                np.diag,
                component_stats,
                input_core_dims=[component_stats.dims],
                output_core_dims=[[self.params.component_axis]],
                dask="allowed",
            )

            # Apply update equation with masking
            update = numerator / M_diag
            A_new = xr.where(mask, A + update, A)
            A_new = xr.where(A_new > 0, A_new, 0)
            if abs((A - A_new).sum() / np.prod(A.shape)) < self.params.tolerance:
                A = A_new
                converged = True
            else:
                A = A_new

        self.footprints_ = A
        self.is_fitted_ = True
        return self

    def transform_one(self, _: None = None) -> Footprints:
        """Return the updated spatial footprints.

        This method returns the updated footprints after the shape optimization
        process has completed.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            Footprints: Updated spatial footprints with optimized shapes.

        Raises:
            NotFittedError: If the transformer hasn't been fitted yet.
        """
        if not self.is_fitted_:
            raise NotFittedError

        return self.footprints_
