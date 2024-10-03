from dask import delayed
import dask


class Preprocessor:
    @delayed
    def process_frame(self, frame):
        return frame.mean()

    def process_video_in_batches(self, video_path, batch_size):
        for batch in self.read_video_frames(video_path, batch_size):
            tasks = [self.process_frame(frame) for frame in batch]
            yield tasks


# batches
video_path = "large_video.mp4"
batch_size = 100  # Adjust based on memory constraints
all_results = []

for tasks in Preprocessor().process_video_in_batches(video_path, batch_size):
    results = dask.compute(*tasks)  # Compute results for each batch
    all_results.extend(results)
