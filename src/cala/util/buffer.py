from pydantic import BaseModel, ConfigDict, Field
from collections import deque

import xarray as xr

from cala.models import AXIS


class Buffer(BaseModel):
    size: int
    frames: deque[xr.DataArray] = Field(default_factory=deque)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def add(self, frames: xr.DataArray) -> None:
        """
        Add a new frame to the ring buffer.
        """
        frames = (
            [frame for frame in frames.transpose(AXIS.frames_dim, ...)]
            if frames.ndim == 3
            else [frames]
        )

        for frame in frames:
            self.frames.append(frame)

            if len(self.frames) > self.size:
                self.frames.popleft()

    def get_latest(self, n: int = 1) -> xr.DataArray:
        """Get n most recent frames.

        Returns:
            xr.DataArray: A 3D array containing the stacked frames.
        """
        if not self.is_ready(n):
            raise ValueError("Buffer does not have enough frames.")

        if n == 1:
            return self.frames[-1]

        return xr.concat(list(self.frames)[-n:], dim=AXIS.frames_dim)

    def get_earliest(self, n: int = 1) -> xr.DataArray:
        """Get n earliest frames.

        Returns:
            xr.DataArray: A 3D array containing the stacked frames.
        """
        if not self.is_ready(n):
            raise ValueError("Buffer does not have enough frames.")

        if n == 1:
            return self.frames[0]

        return xr.concat(list(self.frames)[:n], dim=AXIS.frames_dim)

    def is_ready(self, num_frames: int) -> bool:
        """Check if buffer has enough frames."""
        return len(self.frames) >= num_frames

    def clear(self) -> None:
        self.frames.clear()
        return None
