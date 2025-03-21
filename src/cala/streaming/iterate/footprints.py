from dataclasses import dataclass
from typing import Self, List

import numpy as np
import xarray as xr
from river.base import SupervisedTransformer
from sklearn.exceptions import NotFittedError

from cala.streaming.core import Parameters
from cala.streaming.stores.common import Footprints
from cala.streaming.stores.odl import PixelStats, ComponentStats


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
        components_to_update: List[int],
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
            components_to_update (List[int]): List of component indices to update.

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Get working copies of the arrays
        A = footprints.values.copy()
        W = pixel_stats.values
        M = component_stats.values

        # Step 1: Initialize iteration counter
        iter_count = 0

        # Steps 2-8: Main iteration loop
        while iter_count < self.params.max_iterations:
            # Step 3: Loop over components to be updated
            for i in components_to_update:
                # Step 4: Find pixels where component i can be non-zero
                p = np.where(A[:, i] > 0)[0]

                # Step 5: Update footprint values using the update equation
                numerator = W[p, i] - np.sum(A[p, :] * M[i, :], axis=1)
                A[p, i] = np.maximum(A[p, i] + numerator / M[i, i], 0)

            # Step 7: Increment iteration counter
            iter_count += 1

        # Create updated xarray DataArray with same coordinates/dimensions
        self.footprints_ = xr.DataArray(
            A, dims=footprints.dims, coords=footprints.coords
        )

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
