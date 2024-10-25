from typing import List, Iterable, Generator, Literal
from pydantic import validate_arguments
from dask import delayed, compute
from numpydantic import NDArray
import numpy as np

"""
Preprocessing
    * downsampling
            subset, mean
    * calculate chunk
    * glow removal
    * denoise
    * background removal
            This step attempts to estimate background (everything except the fluorescent signal of in-focus cells) and subtracts it from the frame.
            By default we use a morphological tophat operation to estimate the background from each frame:
            First, a [disk element](http://scikit-image.org/docs/dev/api/skimage.morphology.html#disk) with a radius of `wnd` is created.
            Then, a [morphological erosion](https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm) using the disk element is applied to each frame, which eats away any bright "features" that are smaller than the disk element.
            Subsequently, a [morphological dilation](https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm) is applied to the "eroded" image, which in theory undoes the erosion except the bright "features" that were completely eaten away.
            The overall effect of this process is to remove any bright feature that is smaller than a disk with radius `wnd`.
            Thus, when setting `wnd` to the **largest** expected radius of cell, this process can give us a good estimation of the background.
            Then finally the estimated background is subtracted from each frame.
"""


class Preprocessor:

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
    @validate_arguments
    def downsample(
        self,
        chunk: NDArray,
        mode: Literal["mean", "subset"],
        x_downsample: int = 1,
        y_downsample: int = 1,
        t_downsample: int = 1,
    ) -> NDArray:
        pool_size = (t_downsample, x_downsample, y_downsample)
        chunk = self.pad_input(chunk, pool_size)

        if mode == "mean":
            input_shape = chunk.shape
            strides = pool_size  # Strides are equal to pool sizes for each axis
            output_shape = tuple(
                input_shape[i] // pool_size[i] for i in range(len(input_shape))
            )
            output = np.zeros(output_shape)

            # Create indices for each dimension
            indices = [
                range(0, input_shape[i], pool_size[i]) for i in range(len(input_shape))
            ]

            # Use numpy.meshgrid to create a grid of indices
            grid = np.meshgrid(*indices, indexing="ij")
            grid_shape = grid[0].shape
            flat_indices = [g.flatten() for g in grid]

            for idx in zip(*flat_indices):
                slices = tuple(
                    slice(idx[i], idx[i] + pool_size[i])
                    for i in range(len(input_shape))
                )
                window = chunk[slices]
                output_idx = tuple(
                    idx[i] // pool_size[i] for i in range(len(input_shape))
                )
                output[output_idx] = np.mean(window)

            return output
        elif mode == "subset":
            return chunk[::t_downsample, ::x_downsample, ::y_downsample]
        else:
            raise ValueError("mode must be either 'mean' or 'subset'")

    @delayed
    @validate_arguments
    def preprocess_batch_2(self, batch: NDArray) -> NDArray:
        return batch * 2

    @staticmethod
    def yield_in_batches(
        video: NDArray, batch_size: int = 100
    ) -> Generator[Iterable[NDArray], None, None]:
        """Yield successive batches from the video array."""
        for i in range(0, len(video), batch_size):
            yield video[i : i + batch_size]

    def process_video_in_batches(
        self,
        video: NDArray,
        batch_size: int,
        downsample_mode: Literal["mean", "subset"],
        x_downsample: int = 1,
        y_downsample: int = 1,
        t_downsample: int = 1,
    ) -> List[NDArray]:
        """
        Process video frames in parallel batches using Dask.

        Args:
            video (NDArray): The video data as a NumPy array.
            batch_size (int): Number of frames per batch.
            downsample_mode (str): mean or subset
            x_downsample (int):
            y_downsample (int):
            t_downsample (int):

        Returns:
            List[NDArray]: List of processed batches.
        """
        delayed_tasks = []

        for batch in self.yield_in_batches(video, batch_size):
            processed = self.downsample(
                batch,
                mode=downsample_mode,
                x_downsample=x_downsample,
                y_downsample=y_downsample,
                t_downsample=t_downsample,
            )
            processed = self.preprocess_batch_2(processed)
            delayed_tasks.append(processed)

        processed_batches = compute(*delayed_tasks)

        return processed_batches
