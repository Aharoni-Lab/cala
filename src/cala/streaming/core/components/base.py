from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum, auto
import numpy as np
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
    old_footprint: Optional[sparse.csr_matrix] = None
    """Previous footprint"""
    old_time_trace: Optional[np.ndarray] = None
    """Previous time trace"""


@dataclass
class FluorescentObject(ABC):
    """Base class for any fluorescent object detected."""

    _id: int = field(init=False)
    """Unique identifier for the object"""

    _footprint: sparse.csr_matrix
    """2D sparse matrix representing spatial distribution"""
    time_trace: np.ndarray
    """1D array of fluorescence intensity over time"""

    confidence_level: float
    """Confidence in object detection/separation"""

    overlapping_objects: Set["FluorescentObject"] = field(default_factory=set)
    """Objects with overlapping footprints"""

    last_update: ComponentUpdate = field(init=False)
    """Last update to the object"""

    def __init__(
        self,
        footprint: np.ndarray,
        time_trace: np.ndarray,
        confidence_level: float,
        overlapping_objects: Optional[Set["FluorescentObject"]] = None,
    ):
        """Initialize the object with a dense footprint that will be converted to sparse."""
        self._footprint = sparse.csr_matrix(footprint)
        self.time_trace = time_trace
        self.confidence_level = confidence_level
        self.overlapping_objects = (
            set() if overlapping_objects is None else overlapping_objects
        )
        self._id = id(self)

    @property
    def footprint(self) -> sparse.csr_matrix:
        """Get the sparse footprint matrix."""
        return self._footprint

    @footprint.setter
    def footprint(self, value: np.ndarray | sparse.spmatrix) -> None:
        """Set the footprint, converting to sparse if needed."""
        self._footprint = sparse.csr_matrix(value)

    def update_footprint(self, footprint: np.ndarray) -> None:
        """Update the footprint of the object."""
        self.last_update = ComponentUpdate(
            update_type=UpdateType.MODIFIED, old_footprint=self._footprint
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
