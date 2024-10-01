from pydantic import BaseModel
from typing import List
from pathlib import Path
import numpy as np
import zarr
from numcodecs import blosc
import av
from tqdm import tqdm
from numpydantic.interface.video import VideoProxy


class VideoMetadata(BaseModel):
    num_frames: int
    height: int
    width: int
    channels: int


class DataIO:
    def __init__(
        self,
        video_directory: Path,
        video_files: List[str],
    ) -> None:
        self.video_directory = video_directory
        self.video_files = video_files
        self.compressor = blosc.Blosc(cname="zstd", clevel=3, shuffle=2)

    @staticmethod
    def _load_video_metadata(video_path: Path):
        video = VideoProxy(path=video_path)
        return video

    @property
    def metadata(self) -> VideoMetadata:
        num_frames = 0
        for idx, video_file in enumerate(self.video_files):
            video_path = self.video_directory.joinpath(video_file)
            video = self._load_video_metadata(video_path)
            num_frames += video.n_frames
            if idx > 0:
                if (height, width, channels) != video.shape[1:4]:
                    raise Exception(
                        f"Frame dimensions of {video_path} do not match to its predecessor."
                    )
            height = video.shape[1]
            width = video.shape[2]
            channels = video.shape[3]

        return VideoMetadata(
            num_frames=num_frames, height=height, width=width, channels=channels
        )

    def save_data(
        self,
        save_directory: Path,
        save_name: str,
        chunk_size: int = 1024,
        dtype: np.dtype = np.uint8,
    ) -> None:
        n_frames = self.metadata.num_frames
        width = self.metadata.width
        height = self.metadata.height

        zarr_store = zarr.open(
            store=f"{save_directory / save_name}.zarr",
            mode="w",
            shape=(n_frames, height, width),
            chunks=(chunk_size, height, width),
            dtype=dtype,
            compressor=self.compressor,
        )

        for video_file in tqdm(self.video_files):
            container = av.open(self.video_directory.joinpath(video_file))
            video_stream = container.streams.video[0]

            for packet in container.demux(video_stream):
                for frame in packet.decode():
                    frame_array = frame.to_ndarray(format="gray")
                    zarr_store.append([frame_array])

            container.close()


def main():
    data_io = DataIO(
        video_directory=Path("/Users/raymond/Documents/GitHub/cala/data"),
        video_files=["msCam1.avi", "msCam2.avi", "msCam3.avi"],
    )

    print(data_io.metadata)

    data_io.save_data(
        save_directory=Path("/Users/raymond/Documents/GitHub/cala/data"),
        save_name="temp",
    )


if __name__ == "__main__":
    main()
