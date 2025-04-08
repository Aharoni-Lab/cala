from dataclasses import dataclass
from typing import Self

import sparse
import xarray as xr
from river.base import SupervisedTransformer
from sklearn.exceptions import NotFittedError

from cala.streaming.core import Axis, Parameters
from cala.streaming.stores.common import Footprints
from cala.streaming.stores.odl import Overlaps


@dataclass
class OverlapsUpdaterParams(Parameters, Axis):
    """Parameters for component statistics updates.

    This class defines the configuration parameters needed for updating
    component-wise statistics matrices.
    """

    def validate(self) -> None:
        """Validate parameter configurations.

        This implementation has no parameters to validate, but the method
        is included for consistency with the Parameters interface.
        """
        pass


@dataclass
class OverlapsUpdater(SupervisedTransformer):
    """Updates overlaps matrices using updated footprints.

    This transformer implements the overlaps statistics update.
    Currently it is done in a brute force manner where we take all footprints and recalculate
    overlaps from scratch, since we're not tracking which footprints have undergone boundary changes.
    """

    params: OverlapsUpdaterParams
    """Configuration parameters for the update process."""

    is_fitted_: bool = False
    """Indicator whether the transformer has been fitted."""

    overlaps_: Overlaps = None

    def learn_one(
        self,
        frame: xr.DataArray,
        footprints: Footprints,
    ) -> Self:
        """Update overlaps using current footprints.

        The implementation is identical to initialization, currently.

        Args:
            frame (Frame): Current frame y_t. Unused
                Shape: (height × width)
            footprints (Footprints): Current temporal component c_t.
                Shape: (components)

        Returns:
            Self: The transformer instance for method chaining.
        """

        # Use matrix multiplication with broadcasting to compute overlaps
        data = (
            footprints.dot(
                footprints.rename({self.params.component_axis: f"{self.params.component_axis}'"})
            )
            > 0
        ).astype(int)

        # Create xarray DataArray with sparse data
        data.values = sparse.COO(data.values)
        self.overlaps_ = data.assign_coords(
            {
                self.params.id_coordinates: (
                    self.params.component_axis,
                    footprints.coords[self.params.id_coordinates].values,
                ),
                self.params.type_coordinates: (
                    self.params.component_axis,
                    footprints.coords[self.params.type_coordinates].values,
                ),
            }
        )

        self.is_fitted_ = True
        return self

    def transform_one(self, _: None = None) -> Overlaps:
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

        return self.overlaps_
