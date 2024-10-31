from pathlib import Path
from typing import List, Union

import av
import numpy as np
import zarr
from numcodecs import blosc
from numpydantic import NDArray, Shape
from pydantic import BaseModel
from tqdm import tqdm


class Image(BaseModel):
    array: Union[
        NDArray[Shape["* x, * y"], float],  # noqa: F722,F821
        NDArray[Shape["* x, * y, 3 rgb"], np.uint8],  # noqa: F722,F821
        NDArray[Shape["* t, 1080 y, 1920 x, 3 rgb"], np.uint8],  # noqa: F722,F821
    ]


class VideoMetadata(BaseModel):
    num_frames: int
    height: int
    width: int
    channels: int


class DataIO:
    def __init__(self, video_paths: List[Path]) -> None:
        self.video_paths = video_paths
        self.compressor = blosc.Blosc(cname="zstd", clevel=3, shuffle=2)

    @property
    def metadata(self) -> VideoMetadata:
        num_frames = 0
        height = width = channels = None

        for idx, video_path in enumerate(self.video_paths):
            video = Image(path=video_path)
            num_frames += video.n_frames
            if (idx > 0) and ((height, width, channels) != video.shape[1:]):
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
        data_directory: Path,
        data_name: str,
        chunk_size: int = 1024,
        dtype: np.dtype = np.uint8,
    ) -> None:
        num_frames = self.metadata.num_frames
        width = self.metadata.width
        height = self.metadata.height

        zarr_store = zarr.open(
            store=f"{data_directory / data_name}.zarr",
            mode="w",
            shape=(num_frames, height, width),
            chunks=(chunk_size, height, width),
            dtype=dtype,
            compressor=self.compressor,
        )

        for video_path in tqdm(self.video_paths):
            container = av.open(video_path)
            video_stream = container.streams.video[0]

            for packet in tqdm(container.demux(video_stream), leave=False):
                for idx, frame in enumerate(packet.decode()):
                    frame_array = frame.to_ndarray(format="gray")
                    zarr_store[idx] = frame_array

            container.close()
