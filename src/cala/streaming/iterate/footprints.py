from dataclasses import dataclass
from typing import Self

import numpy as np
import xarray as xr
from river.base import SupervisedTransformer
from scipy.ndimage import binary_dilation
from sklearn.exceptions import NotFittedError

from cala.streaming.composer import Frame
from cala.streaming.core import Parameters
from cala.streaming.stores.common import Footprints
from cala.streaming.stores.odl import ComponentStats, PixelStats


@dataclass
class FootprintsUpdaterParams(Parameters):
    """Parameters for spatial footprint updates.

    This class defines the configuration parameters needed for updating
    spatial footprints of components, including axis names and iteration limits.
    """

    component_axis: str = "components"
    """Name of the dimension representing individual components."""

    spatial_axes: tuple = ("height", "width")
    """Names of the dimensions representing spatial coordinates (height, width)."""

    boundary_expansion_pixels: int | None = None
    """Number of pixels to explore the boundary of the footprint outside of the current footprint."""

    max_iterations: int = 100
    """Maximum number of iterations for shape update convergence."""

    def validate(self):
        """Validate parameter configurations.

        Raises:
            ValueError: If max_iterations is not positive.
        """
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")


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
        frame: Frame,
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

        for _ in range(self.params.max_iterations):
            # Create mask for non-zero pixels per component
            mask = A > 0
            if self.params.boundary_expansion_pixels:
                mask = xr.apply_ufunc(
                    lambda x: binary_dilation(
                        x, iterations=self.params.boundary_expansion_pixels
                    ),
                    mask,
                    input_core_dims=[[*self.params.spatial_axes]],
                    output_core_dims=[[*self.params.spatial_axes]],
                    vectorize=True,
                    dask="allowed",
                )
            # Compute AM product using xarray operations
            # Reshape M to align dimensions for broadcasting
            AM = (A @ M).rename(
                {f"{self.params.component_axis}'": f"{self.params.component_axis}"}
            )
            numerator = pixel_stats - AM

            # Compute update using vectorized operations
            # Expand M diagonal for broadcasting
            M_diag = xr.apply_ufunc(
                np.diag,
                component_stats,
                input_core_dims=[component_stats.dims],
                output_core_dims=[[self.params.component_axis]],
            )

            # Apply update equation with masking
            update = numerator / M_diag
            A = xr.where(mask, A + update, A)
            A = xr.where(A > 0, A, 0)

        self.footprints_ = A
        self.is_fitted_ = True
        return self

    def transform_one(self, _=None) -> Footprints:
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
