from typing import Any

import av
import numpy as np
import xarray as xr
from av.video import VideoStream
from noob import process_method
from pydantic import BaseModel

from cala.config import config
from cala.models import AXIS


class Streamer(BaseModel):
    frame_rate: int
    _stream: VideoStream | None = None
    _container: av.container.OutputContainer | None = None

    def model_post_init(self, context: Any, /) -> None:
        hls_manifest = config.runtime_dir / "stream.m3u8"
        self._container = av.open(
            hls_manifest,
            mode="w",
            format="hls",
            options={
                "hls_time": "2",  # segment_duration_in_seconds
                "hls_list_size": "5",  # num_segments_to_keep
                "hls_flags": "delete_segments",  # Auto-delete old segments
                "hls_segment_filename": str(config.runtime_dir / "stream%d.ts"),
            },
        )
        self._stream = self._container.add_stream("h264", rate=self.frame_rate)
        self._stream.pix_fmt = "yuv420p"

    @process_method
    def save(self, frame: xr.DataArray) -> xr.DataArray:
        frame = frame.astype(np.uint8)
        self._stream.width = frame.sizes[AXIS.width_dim]
        self._stream.height = frame.sizes[AXIS.height_dim]

        try:
            vid_frame = av.VideoFrame.from_ndarray(frame.to_numpy(), format="gray")
            packets = self._stream.encode(vid_frame.reformat(format=self._stream.pix_fmt))
            for packet in packets:
                self._container.mux(packet)

        except Exception as e:
            self._container.close()
            raise Exception(f"VideoStreamError: {e}") from e

        return frame

    def deinit(self) -> None:
        # Flush any remaining packets
        if self._stream:
            for packet in self._stream.encode(None):
                self._container.mux(packet)

        if self._container:
            self._container.close()
