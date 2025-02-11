from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import xarray as xr
from river import base


@dataclass
class GlowRemover(base.Transformer):
    base_brightness_: np.ndarray = field(init=False)
    _learn_count: int = 0
    _transform_count: int = 0

    def learn_one(self, X: xr.DataArray, y=None):
        if not hasattr(self, "base_brightness_"):
            self.base_brightness_ = X.values
        else:
            self.base_brightness_ = np.minimum(X.values, self.base_brightness_)
        self._learn_count += 1
        return self

    def transform_one(self, X: xr.DataArray, y=None):
        self._transform_count += 1
        return X - self.base_brightness_

    def get_info(self) -> Dict:
        """Get information about the current state.

        Returns
        -------
        dict
            Dictionary containing current statistics
        """
        return {
            "base_brightness_": self.base_brightness_,
            "transform_count": self._transform_count,
            "learn_count": self._learn_count,
        }
