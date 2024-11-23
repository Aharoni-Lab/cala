from dataclasses import dataclass
from typing import Self

import dask.array as da
import numpy as np
import xarray as xr
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import lsqr

from .base import BaseDemixer


@dataclass
class TemporalUpdater(BaseDemixer):
    iter_lim: int = 10
    cell_id_axis: str = "cell_id"
    features_dask: csc_matrix = None
    cell_ids_: xr.DataArray = None

    def fit_kernel(self, features: xr.DataArray, X=None) -> None:
        # Store cell IDs
        self.cell_ids_ = features.coords[self.cell_id_axis]

        # Stack and preprocess spatial dimensions of features
        features_stacked = (
            features.stack({self.spatial_axis: self.core_axes})
            .transpose(self.spatial_axis, self.cell_id_axis)
            .data
        )
        self.features_dask = features_stacked.map_blocks(csc_matrix, dtype=csc_matrix)

    def fit(self, X: xr.DataArray, y: xr.DataArray) -> Self:
        self.fit_kernel(X=X, features=y)

        return self

    def transform_kernel(self, X: xr.DataArray, y=None) -> xr.DataArray:
        frames = X.coords[self.iter_axis]
        X_stacked = X.stack(spatial=self.core_axes).transpose(
            self.iter_axis, self.spatial_axis
        )

        weights = xr.apply_ufunc(
            self._sparse_least_squares,
            self.features_dask,
            X_stacked,
            input_core_dims=[
                [self.spatial_axis, self.cell_id_axis],
                [self.iter_axis, self.spatial_axis],
            ],
            output_core_dims=[[self.iter_axis, self.cell_id_axis]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
            kwargs={"iter_lim": self.iter_lim},
        )

        return weights.assign_coords(
            {self.cell_id_axis: self.cell_ids_, self.iter_axis: frames}
        ).transpose(self.cell_id_axis, self.iter_axis)

    def transform(self, X: xr.DataArray, y) -> xr.DataArray:
        if self.features_dask is None or self.cell_ids_ is None:
            raise RuntimeError("The estimator must be fitted before calling transform.")

        return self.transform_kernel(X, y)

    @staticmethod
    def _sparse_least_squares(A: csc_matrix, b: np.ndarray, **kwargs):
        """
        Generalized UFunc for performing sparse least-squares over rows of b.
        """
        return lsqr(A, b.squeeze(), **kwargs)[0]
