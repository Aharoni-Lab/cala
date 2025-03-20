from dataclasses import dataclass
from typing import Self, Tuple

import numpy as np
import xarray as xr
from river.base import SupervisedTransformer
from sklearn.exceptions import NotFittedError

from cala.streaming.core import Parameters
from cala.streaming.stores.odl import PixelStats, ComponentStats


@dataclass
class UpdateStatisticsParams(Parameters):
    """Parameters for sufficient statistics updates.

    This class defines the configuration parameters needed for updating
    pixel-wise and component-wise sufficient statistics matrices.
    """

    component_axis: str = "components"
    """Name of the dimension representing individual components."""

    spatial_axes: tuple = ("height", "width")
    """Names of the dimensions representing spatial coordinates (height, width)."""

    def validate(self):
        """Validate parameter configurations.

        This implementation has no parameters to validate, but the method
        is included for consistency with the Parameters interface.
        """
        pass


@dataclass
class UpdateStatistics(SupervisedTransformer):
    """Updates sufficient statistics matrices using current frame.

    This transformer implements the sufficient statistics update equations:

    W_t = ((t-1)/t)W_{t-1} + (1/t)y_t c_t^T
    M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T

    where:
    - W_t is the pixel-wise sufficient statistics at time t
    - M_t is the component-wise sufficient statistics at time t
    - y_t is the current frame
    - c_t is the current temporal component
    - t is the current timestep
    """

    params: UpdateStatisticsParams
    """Configuration parameters for the update process."""

    pixel_stats_: PixelStats = None
    """Updated pixel-wise sufficient statistics W."""

    component_stats_: ComponentStats = None
    """Updated component-wise sufficient statistics M."""

    is_fitted_: bool = False
    """Indicator whether the transformer has been fitted."""

    def learn_one(
        self,
        frame: xr.DataArray,
        temporal_component: xr.DataArray,
        pixel_stats: PixelStats,
        component_stats: ComponentStats,
        timestep: int,
    ) -> Self:
        """Update sufficient statistics using current frame and component.

        This method implements the update equations for both pixel-wise
        and component-wise sufficient statistics matrices. The updates
        incorporate the current frame and temporal component with
        appropriate time-based scaling.

        Args:
            frame (xr.DataArray): Current frame y_t.
                Shape: (height × width)
            temporal_component (xr.DataArray): Current temporal component c_t.
                Shape: (components)
            pixel_stats (PixelStats): Current pixel-wise statistics W_{t-1}.
                Shape: (pixels × components)
            component_stats (ComponentStats): Current component-wise statistics M_{t-1}.
                Shape: (components × components)
            timestep (int): Current timestep t.

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Compute scaling factors
        prev_scale = (timestep - 1) / timestep
        new_scale = 1 / timestep

        # Flatten spatial dimensions of frame
        y_t = frame.values.reshape(-1)
        c_t = temporal_component.values

        # Update pixel-wise statistics W_t
        # W_t = ((t-1)/t)W_{t-1} + (1/t)y_t c_t^T
        W_update = prev_scale * pixel_stats.values + new_scale * np.outer(y_t, c_t)

        # Update component-wise statistics M_t
        # M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T
        M_update = prev_scale * component_stats.values + new_scale * np.outer(c_t, c_t)

        # Create updated xarray DataArrays with same coordinates/dimensions
        self.pixel_stats_ = xr.DataArray(
            W_update, dims=pixel_stats.dims, coords=pixel_stats.coords
        )
        self.component_stats_ = xr.DataArray(
            M_update, dims=component_stats.dims, coords=component_stats.coords
        )

        self.is_fitted_ = True
        return self

    def transform_one(self, _=None) -> Tuple[PixelStats, ComponentStats]:
        """Return the updated sufficient statistics matrices.

        This method returns both updated statistics matrices after the
        update process has completed.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            tuple:
                - PixelStats: Updated pixel-wise sufficient statistics W_t
                - ComponentStats: Updated component-wise sufficient statistics M_t

        Raises:
            NotFittedError: If the transformer hasn't been fitted yet.
        """
        if not self.is_fitted_:
            raise NotFittedError

        return self.pixel_stats_, self.component_stats_
