from dataclasses import dataclass
from typing import Self

import dask.array as da
import numpy as np
import sparse
import xarray as xr
from dask import delayed
from scipy.sparse.linalg import lsqr

from .base import BaseDemixer


@dataclass
class TemporalUpdater(BaseDemixer):
    iter_lim: int = 10
    feature_id_axis: str = "unit_id"
    cell_ids_: xr.DataArray = None
    features_sparse_dask_: da = None
    features_shape_: np.ndarray = None

    def fit_kernel(self, features: xr.DataArray, X=None):
        self.cell_ids_ = features.coords[self.feature_id_axis]

        features_stacked = features.stack(
            {self.spatial_axis: self.core_axes}
        ).transpose(self.spatial_axis, self.feature_id_axis)
        features = features_stacked.data

        def to_sparse_coo(block):
            return sparse.COO(np.array(block))

        features_sparse_dask = features.map_blocks(to_sparse_coo, dtype=sparse.COO)

        self.features_sparse_dask_ = features_sparse_dask
        self.features_shape_ = features.shape

        return self

    def fit(self, X: xr.DataArray, y: xr.DataArray) -> Self:
        self.fit_kernel(X=X, features=y)

        return self

    def transform_kernel(self, X: xr.DataArray, y=None) -> xr.DataArray:
        frames = X.coords[self.iter_axis]
        X_stacked = X.stack({self.spatial_axis: self.core_axes}).transpose(
            self.iter_axis, self.spatial_axis
        )
        X_data = X_stacked.data

        # Define the function to solve least squares for a block of frames
        def solve_least_squares_block(frames_block, feature_block):
            num_frames_block = frames_block.shape[0]
            num_units = feature_block.shape[1]
            weight_block = np.zeros((num_frames_block, num_units))
            feature_block = feature_block.tocsr()
            for i in range(num_frames_block):
                b_i = frames_block[i, :]
                x_i = lsqr(feature_block, b_i, iter_lim=self.iter_lim)[0]
                weight_block[i, :] = x_i
            return weight_block

        # Get the chunks of X_data and features_sparse_dask_
        X_blocks = X_data.to_delayed().flatten()
        feature_blocks = self.features_sparse_dask_.to_delayed().flatten()

        # Ensure that feature_blocks and X_blocks are aligned
        if len(feature_blocks) != len(X_blocks):
            # Adjust the chunks to align them
            feature_blocks = [
                self.features_sparse_dask_.to_delayed().flatten()[0]
            ] * len(X_blocks)

        weight_blocks = []

        for frames_block_delayed, feature_block_delayed in zip(
            X_blocks, feature_blocks
        ):
            weight_block_delayed = delayed(solve_least_squares_block)(
                frames_block_delayed, feature_block_delayed
            )
            num_frames_block = frames_block_delayed.shape[0]
            num_units = self.features_shape_[1]
            weight_block = da.from_delayed(
                weight_block_delayed,
                shape=(num_frames_block, num_units),
                dtype=np.float64,
            )
            weight_blocks.append(weight_block)

        # Concatenate the computed blocks along the frame dimension
        weights_data = da.concatenate(weight_blocks, axis=0)

        # Construct the output DataArray weights
        weights = xr.DataArray(
            weights_data,
            dims=[self.iter_axis, self.feature_id_axis],
            coords={self.feature_id_axis: self.feature_id_axis, self.iter_axis: frames},
        ).transpose(self.feature_id_axis, self.iter_axis)

        return weights

    def transform(self, X: xr.DataArray, y) -> xr.DataArray:
        if self.features_sparse_dask_ is None:
            raise RuntimeError("The estimator must be fitted before calling transform.")

        return self.transform_kernel(X, y)
