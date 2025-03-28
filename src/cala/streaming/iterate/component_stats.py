from dataclasses import dataclass
from typing import Self

import numpy as np
import xarray as xr
from river.base import SupervisedTransformer
from sklearn.exceptions import NotFittedError

from cala.streaming.composer import Frame
from cala.streaming.core import Parameters, Axis
from cala.streaming.stores.common import Traces
from cala.streaming.stores.odl import ComponentStats


@dataclass
class ComponentStatsUpdaterParams(Parameters, Axis):
    """Parameters for component statistics updates.

    This class defines the configuration parameters needed for updating
    component-wise statistics matrices.
    """

    def validate(self):
        """Validate parameter configurations.

        This implementation has no parameters to validate, but the method
        is included for consistency with the Parameters interface.
        """
        pass


@dataclass
class ComponentStatsUpdater(SupervisedTransformer):
    """Updates component statistics matrices using current frame.

    This transformer implements the component statistics update equation:

    M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T

    where:
    - M_t is the component-wise sufficient statistics at time t
    - y_t is the current frame
    - c_t is the current temporal component
    - t is the current timestep
    """

    params: ComponentStatsUpdaterParams
    """Configuration parameters for the update process."""

    component_stats_: ComponentStats = None
    """Updated component-wise sufficient statistics M."""

    is_fitted_: bool = False
    """Indicator whether the transformer has been fitted."""

    def learn_one(
        self,
        frame: Frame,
        traces: Traces,
        component_stats: ComponentStats,
    ) -> Self:
        """Update component statistics using current frame and component.

        This method implements the update equations for component-wise
        statistics matrices. The updates incorporate the temporal component
        with appropriate time-based scaling.

        Args:
            frame (Frame): Current frame y_t.
                Shape: (height × width)
            traces (Traces): Current temporal component c_t.
                Shape: (components)
            component_stats (ComponentStats): Current component-wise statistics M_{t-1}.
                Shape: (components × components)

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Compute scaling factors
        frame_idx = frame.index + 1
        prev_scale = (frame_idx - 1) / frame_idx
        new_scale = 1 / frame_idx

        # New frame traces
        c_t = traces.isel({self.params.frames_axis: -1})

        # Update component-wise statistics M_t
        # M_t = ((t-1)/t)M_{t-1} + (1/t)c_t c_t^T
        new_corr = xr.DataArray(
            np.outer(c_t, c_t), dims=(c_t.dims[0], f"{c_t.dims[0]}'"), coords=c_t.coords
        )
        M_update = prev_scale * component_stats + new_scale * new_corr

        # Create updated xarray DataArrays with same coordinates/dimensions
        self.component_stats_ = M_update

        self.is_fitted_ = True
        return self

    def transform_one(self, _=None) -> ComponentStats:
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

        return self.component_stats_
