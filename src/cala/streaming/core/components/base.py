from abc import ABC
from dataclasses import dataclass, field
from typing import List, Optional, Set
from enum import Enum, auto
import numpy as np


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
    old_footprint: Optional[np.ndarray] = None
    """Previous footprint"""
    old_time_trace: Optional[np.ndarray] = None
    """Previous time trace"""


@dataclass
class FluorescentObject(ABC):
    """Base class for any fluorescent object detected."""

    _id: int = field(init=False)
    """Unique identifier for the object"""

    footprint: np.ndarray
    """2D array representing spatial distribution"""
    time_trace: np.ndarray
    """1D array of fluorescence intensity over time"""

    confidence_level: float
    """Confidence in object detection/separation"""

    overlapping_objects: Set["FluorescentObject"] = field(default_factory=set)
    """Objects with overlapping footprints"""

    last_update: ComponentUpdate = field(init=False)
    """Last update to the object"""

    def __post_init__(self):
        """Initialize the object after creation."""
        self.overlapping_objects = (
            set() if self.overlapping_objects is None else set(self.overlapping_objects)
        )
        # Initialize component ID
        self._id = id(self)

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
