from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from cala.models import Frame


@dataclass
class GlowRemover:
    base_brightness_: np.ndarray = field(init=False)
    _learn_count: int = 0

    def process(self, frame: Frame) -> Frame:
        frame = frame.array

        if not hasattr(self, "base_brightness_"):
            self.base_brightness_ = frame.values

        self.base_brightness_ = np.minimum(frame.values, self.base_brightness_)
        self._learn_count += 1

        return Frame(
            array=xr.DataArray(frame - self.base_brightness_, dims=frame.dims, coords=frame.coords)
        )

    def get_info(self) -> dict:
        """Get information about the current state.

        Returns
        -------
        dict
            Dictionary containing current statistics
        """
        return {
            "base_brightness_": self.base_brightness_,
            "learn_count": self._learn_count,
        }
