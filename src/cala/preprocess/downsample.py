from typing import List, Iterable, Generator, Literal, Tuple

import numpy as np
from dask import delayed, compute
from numpydantic import NDArray
from sklearn.base import BaseEstimator, TransformerMixin


@delayed
def subset_downsample(nd_array: NDArray, pool_size: Tuple[int, ...]) -> NDArray:
    if nd_array.ndim != len(pool_size):
        raise ValueError(
            "Length of pool_size must match the number of dimensions in nd_array."
        )
    # Create a tuple of slice objects for each dimension
    slices = tuple(slice(None, None, stride) for stride in pool_size)
    # Apply the slices to downsample the array
    return nd_array[slices]


@delayed
def mean_downsample(nd_array: NDArray, pool_size: Tuple[int, ...]) -> NDArray:
    if nd_array.ndim != len(pool_size):
        raise ValueError(
            "Length of pool_size must match the number of dimensions in nd_array."
        )
    input_shape = nd_array.shape
    output_shape = tuple(
        input_shape[i] // pool_size[i] for i in range(len(input_shape))
    )
    output = np.zeros(output_shape)

    # Create indices for each dimension
    indices = [range(0, input_shape[i], pool_size[i]) for i in range(len(input_shape))]

    # Use numpy.meshgrid to create a grid of indices
    grid = np.meshgrid(*indices, indexing="ij")
    flat_indices = [g.flatten() for g in grid]

    for idx in zip(*flat_indices):
        slices = tuple(
            slice(idx[i], idx[i] + pool_size[i]) for i in range(len(input_shape))
        )
        window = nd_array[slices]
        output_idx = tuple(idx[i] // pool_size[i] for i in range(len(input_shape)))
        output[output_idx] = np.mean(window)

    return output


class Downsampler(BaseEstimator, TransformerMixin):
    methods = {"mean": mean_downsample, "subset": subset_downsample}

    def __init__(
        self,
        method: Literal["mean", "subset"] = "mean",
        x_stride: int = 1,
        y_stride: int = 1,
        t_stride: int = 1,
        batch_size: int = 100,
    ):
        if method not in self.methods:
            raise ValueError(
                f"downsample method '{method}' not understood. "
                f"Available methods are: {', '.join(self.methods.keys())}"
            )
        self.method = method
        self.x_stride = x_stride
        self.y_stride = y_stride
        self.t_stride = t_stride
        self.batch_size = batch_size

    @property
    def pool_size(self):
        return self.t_stride, self.x_stride, self.y_stride

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        """
        Args:
            X (NDArray): The video data.

        Returns:
            List[NDArray]: List of processed batches.
        """
        delayed_tasks = []

        for batch in self._yield_in_batches(X, self.batch_size):
            processed = self.downsample(batch)
            delayed_tasks.append(processed)

        processed_batches = compute(*delayed_tasks)

        return processed_batches

    @staticmethod
    def pad_input(input_array, pool_size):
        pad_width = []
        for i, size in enumerate(pool_size):
            total_pad = (size - (input_array.shape[i] % size)) % size
            pad_before = total_pad // 2
            pad_after = total_pad - pad_before
            pad_width.append((pad_before, pad_after))
        return np.pad(input_array, pad_width, mode="constant", constant_values=0)

    @delayed
    def downsample(
        self,
        chunk: NDArray,
    ) -> NDArray:
        chunk = self.pad_input(chunk, self.pool_size)
        return self.methods[self.method](chunk, self.pool_size)

    @staticmethod
    def _yield_in_batches(
        video: NDArray, batch_size
    ) -> Generator[Iterable[NDArray], None, None]:
        """Yield successive batches from the video array."""
        for i in range(0, len(video), batch_size):
            yield video[i : i + batch_size]
