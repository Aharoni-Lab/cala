from typing import Optional, Set

import numpy as np
import zarr
from scipy import sparse

from .base import FluorescentObject
from .neuron import Neuron


class Background(FluorescentObject):
    """Class representing background components in calcium imaging."""

    def __init__(
        self,
        footprint: np.ndarray | sparse.csr_matrix,
        time_trace: np.ndarray | zarr.Array,
        confidence_level: Optional[float] = 0.0,
        overlapping_objects: Optional[Set["FluorescentObject"]] = None,
        background_type: str = "neuropil",
    ):
        """Initialize a Background component.

        Args:
            footprint: 2D array representing spatial distribution. Will be converted to sparse matrix.
            time_trace: 1D array of fluorescence intensity. Will be converted to zarr array.
            confidence_level: Confidence in object detection/separation.
            overlapping_objects: Set of objects with overlapping footprints.
            background_type: Type of background (neuropil, blood vessel, etc.).
        """
        super().__init__(footprint, time_trace, confidence_level, overlapping_objects)
        self._background_type = background_type

    def estimate_contamination(self, neuron: Neuron) -> float:
        """Estimate contamination of this background component on a neuron."""
        raise NotImplementedError
