from abc import ABC
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional


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
    last_update_frame_idx: Optional[int] = None
    """Frame index of last update"""


@dataclass
class FluorescentObject(ABC):
    """Base class for any fluorescent object detected."""

    detected_frame_idx: Optional[int] = None
    """Frame index of object detection"""
    confidence_level: Optional[float] = None
    """Confidence in object detection/separation"""
    last_update: Optional[ComponentUpdate] = None
    """Last update to the object"""

    def __post_init__(self):
        self._id = id(self)
        self._mark_update(UpdateType.ADDED, self.detected_frame_idx)

    @property
    def id(self) -> int:
        """ID of the object."""
        return self._id

    def _mark_update(
        self, update_type: UpdateType, frame_idx: Optional[int] = None
    ) -> None:
        """Helper method to mark an update on the object."""
        self.last_update = ComponentUpdate(
            update_type=update_type, last_update_frame_idx=frame_idx
        )

    def update_confidence_level(
        self, confidence_level: float, frame_idx: Optional[int] = None
    ) -> None:
        """Update the confidence level of the object."""
        self.confidence_level = confidence_level
        self._mark_update(UpdateType.MODIFIED, frame_idx)
