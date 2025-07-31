import os
from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np
import xarray as xr

from cala.models.params import Params


@dataclass
class FrameStreamerParams(Params):
    frame_rate: int
    stream_dir: Path
    playlist_name: str = "stream.m3u8"
    segment_filename: str = "stream%d.ts"
    segment_duration_in_seconds: int = 2
    num_segments_to_keep: int = 5

    def validate(self) -> None:
        try:
            self.stream_dir = Path(self.stream_dir).resolve()
            if not self.stream_dir.is_dir():
                self.stream_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            raise ValueError(f"Invalid stream directory: {e}") from e


@dataclass
class FrameStreamer:
    params: FrameStreamerParams
    _comparison_frame: xr.DataArray | None = None
    _container: av.container.OutputContainer | None = None

    def __post_init__(self):

        self.hls_manifest = os.path.join(self.params.stream_dir, self.params.playlist_name)
        # Create output container for HLS
        self._container = av.open(
            self.hls_manifest,
            mode="w",
            format="hls",
            options={
                "hls_time": str(self.params.segment_duration_in_seconds),
                "hls_list_size": str(self.params.num_segments_to_keep),
                "hls_flags": "delete_segments",  # Auto-delete old segments
                # "hls_playlist_type": "vod",
                # "hls_start_number_source": "datetime",
                # "strftime": "1",
                # "use_localtime": "1",
                "hls_segment_filename": str(
                    Path(self.params.stream_dir) / self.params.segment_filename
                ),
            },
        )
        # Create a video stream
        self.stream = self._container.add_stream("h264", rate=self.params.frame_rate)
        self.stream.pix_fmt = "yuv420p"

    def learn_one(self, frame: xr.DataArray) -> None:
        if frame is not None:
            self._comparison_frame = frame

    def transform_one(self, frame: xr.DataArray) -> xr.DataArray:
        if self._comparison_frame is not None:
            try:
                frame = xr.concat([self._comparison_frame, frame.astype(np.uint8)], dim="width")
            except xr.AlignmentError:
                frame = frame.astype(np.uint8)

        self.stream.width = frame.sizes["width"]
        self.stream.height = frame.sizes["height"]

        try:
            # Create frame
            frame_sample = av.VideoFrame.from_ndarray(frame.to_numpy(), format="gray").reformat(
                format="yuv420p"
            )

            # Encode and write the packet
            packets = self.stream.encode(frame_sample)
            for packet in packets:
                self._container.mux(packet)

        except Exception as e:
            self._container.close()
            raise Exception(f"VideoStreamError: {e}") from e

        return frame

    def close(self) -> None:
        # Flush any remaining packets
        if self.stream:
            for packet in self.stream.encode(None):  # Flush encoder
                self._container.mux(packet)

        # Close the container
        if self._container:
            self._container.close()
