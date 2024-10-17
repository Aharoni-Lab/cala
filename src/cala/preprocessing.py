from typing import List, Iterable, Generator

from dask import delayed, compute
from numpydantic import NDArray


class Preprocessor:
    @delayed
    def preprocess_batch_1(self, batch: NDArray) -> NDArray:
        return batch / 255.0

    @delayed
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
        self, video: NDArray, batch_size: int = 100
    ) -> List[NDArray]:
        """
        Process video frames in parallel batches using Dask.

        Args:
            video (NDArray): The video data as a NumPy array.
            batch_size (int): Number of frames per batch.

        Returns:
            List[NDArray]: List of processed batches.
        """
        delayed_tasks = []

        for batch in self.yield_in_batches(video, batch_size):
            processed = self.preprocess_batch_1(batch)
            processed = self.preprocess_batch_2(processed)
            delayed_tasks.append(processed)

        processed_batches = compute(*delayed_tasks)

        return processed_batches
