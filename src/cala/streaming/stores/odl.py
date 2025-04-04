from typing import Annotated
from uuid import UUID

import numpy as np
from scipy.sparse.csgraph import connected_components
from xarray import DataArray

from cala.streaming.core import ObservableStore


# pixels x components
class PixelStatStore(ObservableStore):
    """Storage for pixel-component statistics.

    This class stores the correlation statistics between each pixel and each component,
    representing how well each pixel's temporal activity aligns with component traces.

    Shape: (pixels × components)
    - pixels: Flattened spatial dimensions (height × width)
    - components: Individual neural and background components

    The values represent normalized correlation coefficients between pixel
    time series and component temporal traces.
    """

    def update(self, data: DataArray) -> None:
        self.warehouse = data


PixelStats = Annotated[DataArray, PixelStatStore]


# components x components
class ComponentStatStore(ObservableStore):
    """Storage for component-component correlation statistics.

    This class stores the correlation matrix between all components,
    representing the temporal relationships between different neural
    and background components.

    Shape: (components × components)
    - Each element (i,j) represents the normalized correlation between
      the temporal traces of components i and j.
    """

    def update(self, data: DataArray) -> None:
        # idt streaming frame index to here makes sense.
        # the actual update calculation should thus happen in update transformer
        self.warehouse = data


ComponentStats = Annotated[DataArray, ComponentStatStore]


class ResidualStore(ObservableStore):
    """Storage for residual signals.

    This class stores the unexplained variance in the data after accounting
    for all identified components. The residual represents signals that
    cannot be explained by the current set of components.

    Shape: (height × width × frames)
    - Maintains spatial structure of the original data
    - Contains only recent frames as specified by buffer length
    - Values represent fluorescence intensity differences between
      original data and component reconstructions
    """

    def update(self, data: DataArray) -> None:
        self.warehouse = data


Residuals = Annotated[DataArray, ResidualStore]


class OverlapStore(ObservableStore):
    """Storage for spatially overlapping component groups.

    This class stores information about which components share spatial overlap
    in their footprints, represented as a sparse matrix for efficiency.

    Shape: (components × components), sparse
    - Non-zero elements indicate spatial overlap between components
    - Sparse representation saves memory for sparse overlap patterns

    Reference:
        https://docs.xarray.dev/en/latest/user-guide/duckarrays.html
    """

    @property
    def labels(self) -> np.ndarray:
        _, labels = connected_components(csgraph=self.warehouse, directed=False, return_labels=True)
        return labels

    @property
    def _ids(self) -> np.ndarray:
        return self.warehouse.coords["id_"].values

    @property
    def groups(self) -> list[list[UUID]]:
        return [list(self._ids[self.labels == label]) for label in np.unique(self.labels)]

    def update(self, data: DataArray) -> None:
        self.warehouse = data


Overlaps = Annotated[DataArray, OverlapStore]
