from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import xarray as xr

from .base import BaseDetector
from skimage.feature import blob_dog, blob_log


@dataclass
class LaplacianOfGaussian(BaseDetector):
    def fit_kernel(self, X: xr.DataArray) -> List[xr.DataArray]:
        return [
            X.isel({self.iter_axis: indices}).max(dim=self.iter_axis)
            for indices in self.window_indices_
        ]

    def transform_kernel(self, X: xr.DataArray) -> pd.DataFrame:
        blobs = blob_log(
            self.window_projections_, max_sigma=30, num_sigma=10, threshold=0.1
        )
        blobs[:, 3] = blobs[:, 3] * np.sqrt(2)
        return pd.DataFrame(blobs, columns=["proj_idx", "y", "x", "r"])
