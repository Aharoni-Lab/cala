from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name

from cala.assets import Frame


class GlowRemover:
    base_brightness_: np.ndarray = None
    _learn_count: int = 0

    def process(self, frame: Frame) -> A[Frame, Name("frame")]:
        frame = frame.array

        if self.base_brightness_ is None:
            self.base_brightness_ = frame.values

        self.base_brightness_ = np.minimum(frame.values, self.base_brightness_)
        self._learn_count += 1

        return Frame.from_array(
            xr.DataArray(frame - self.base_brightness_, dims=frame.dims, coords=frame.coords)
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
