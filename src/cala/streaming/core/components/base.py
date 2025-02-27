from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Set

import numpy as np
import zarr
from scipy import sparse


class UpdateType(Enum):
    """Types of updates that can occur to a component."""

    MODIFIED = auto()  # Component data was modified
    ADDED = auto()  # Component was added
    REMOVED = auto()  # Component was removed


@dataclass
class ComponentUpdate:
    """Tracks updates to a component."""

    update_type: UpdateType
    """Type of update"""
    old_footprint: Optional[np.ndarray | sparse.csr_matrix] = None
    """Previous footprint"""
    old_time_trace: Optional[np.ndarray | zarr.Array] = None
    """Previous time trace"""


class FluorescentObject(ABC):
    """Base class for any fluorescent object detected."""

    def __init__(
        self,
        footprint: np.ndarray | sparse.csr_matrix,
        time_trace: np.ndarray | zarr.Array,
        confidence_level: Optional[float] = None,
        overlapping_objects: Optional[Set["FluorescentObject"]] = None,
    ):
        """Initialize a FluorescentObject.

        Args:
            footprint: 2D array representing spatial distribution. Will be converted to sparse matrix.
            time_trace: 1D array of fluorescence intensity. Will be converted to zarr array.
            confidence_level: Confidence in object detection/separation.
            overlapping_objects: Set of objects with overlapping footprints.
        """
        self._id = id(self)
        self._footprint = (
            sparse.csr_matrix(footprint)
            if not isinstance(footprint, sparse.csr_matrix)
            else footprint
        )
        self._time_trace = (
            zarr.array(time_trace)
            if not isinstance(time_trace, zarr.Array)
            else time_trace
        )
        self.confidence_level = confidence_level
        self.overlapping_objects = overlapping_objects or set()
        self.last_update = ComponentUpdate(update_type=UpdateType.ADDED)

    @property
    def footprint(self) -> sparse.csr_matrix:
        """Get the footprint as a sparse CSR matrix."""
        return self._footprint

    @footprint.setter
    def footprint(self, value: np.ndarray | sparse.csr_matrix) -> None:
        """Set the footprint, converting to sparse CSR matrix if needed."""
        if not isinstance(value, sparse.csr_matrix):
            value = sparse.csr_matrix(value)
        self.last_update = ComponentUpdate(
            update_type=UpdateType.MODIFIED, old_footprint=self._footprint
        )
        self._footprint = value

    @property
    def time_trace(self) -> zarr.Array:
        """Get the time trace as a zarr array."""
        return self._time_trace

    @time_trace.setter
    def time_trace(self, value: np.ndarray | zarr.Array) -> None:
        """Set the time trace, converting to zarr array if needed."""
        if not isinstance(value, zarr.Array):
            value = zarr.array(value)
        self.last_update = ComponentUpdate(
            update_type=UpdateType.MODIFIED, old_time_trace=self._time_trace
        )
        self._time_trace = value

    def update_footprint(self, footprint: np.ndarray) -> None:
        """Update the footprint of the object."""
        self.last_update = ComponentUpdate(
            update_type=UpdateType.MODIFIED, old_footprint=self.footprint
        )
        self.footprint = footprint

    def update_time_trace(self, time_trace: np.ndarray) -> None:
        """Update the time trace of the object."""
        self.last_update = ComponentUpdate(
            update_type=UpdateType.MODIFIED, old_time_trace=self.time_trace
        )
        self.time_trace = time_trace

    def update_confidence_level(self, confidence_level: float) -> None:
        """Update the confidence level of the object."""
        self.confidence_level = confidence_level

    def update_overlapping_objects(
        self, overlapping_objects: Set["FluorescentObject"]
    ) -> None:
        """Update the overlapping objects of the object."""
        self.overlapping_objects = overlapping_objects
