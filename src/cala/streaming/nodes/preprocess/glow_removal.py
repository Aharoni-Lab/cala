from dataclasses import dataclass, field
from typing import Self

import numpy as np
import xarray as xr
from river import base


@dataclass
class GlowRemover(base.Transformer):
    learning_rate: float = 0.1
    base_brightness_: np.ndarray = field(init=False)
    _learn_count: int = 0

    def __post_init__(self) -> None:
        if not (0 < self.learning_rate <= 1):
            raise ValueError(
                f"Parameter learning_rate must be between 0 and 1. Provided: {self.learning_rate}"
            )

    def learn_one(self, frame: xr.DataArray, y: None = None) -> Self:
        if not hasattr(self, "base_brightness_"):
            self.base_brightness_ = np.zeros_like(frame.values)

        else:
            self.base_brightness_ = np.minimum(frame.values, self.base_brightness_) * min(
                self.learning_rate * self._learn_count, 1
            )
        self._learn_count += 1
        return self

    def transform_one(self, frame: xr.DataArray, y: None = None) -> xr.DataArray:
        return xr.DataArray(frame - self.base_brightness_, dims=frame.dims, coords=frame.coords)

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
