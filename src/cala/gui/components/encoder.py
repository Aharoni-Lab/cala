from typing import Any, Literal

import av
import numpy as np
from av.video import VideoStream
from noob.node import Node

from cala.assets import AXIS
from cala.assets.assets import Frame
from cala.config import config
from cala.util import clear_dir


class EncodingError(Exception):
    def __str__(self) -> str:
        return "Encoding failed."


class Encoder(Node):
    frame_rate: int

    color: Literal["gray", "rgb24"] = "gray"
    _stream: VideoStream | None = None
    _container: av.container.OutputContainer | None = None

    def model_post_init(self, context: Any, /) -> None:
        encode_dir = config.runtime_dir / self.id
        encode_dir.mkdir(parents=True, exist_ok=True)
        clear_dir(encode_dir)
        hls_manifest = encode_dir / "stream.m3u8"
        self._container = av.open(
            hls_manifest,
            mode="w",
            format="hls",
            options={
                "hls_time": "2",  # segment_duration_in_seconds
                "hls_list_size": "5",  # num_segments_to_keep
                "hls_flags": "delete_segments",  # Auto-delete old segments
                "hls_segment_filename": str(encode_dir / "stream%d.ts"),
            },
        )
        self._stream = self._container.add_stream("h264", rate=self.frame_rate)
        self._stream.pix_fmt = "yuv420p"

    def process(self, frame: Frame) -> None:
        frame = frame.array.astype(np.uint8)
        self._stream.width = frame.sizes[AXIS.width_dim]
        self._stream.height = frame.sizes[AXIS.height_dim]

        try:
            vid_frame = av.VideoFrame.from_ndarray(frame.to_numpy(), format=self.color)
            packets = self._stream.encode(vid_frame.reformat(format=self._stream.pix_fmt))
            for packet in packets:
                self._container.mux(packet)

        except EncodingWarning as e:
            self._container.close()
            raise EncodingError() from e

    def deinit(self) -> None:
        # Flush any remaining packets
        if self._stream:
            for packet in self._stream.encode(None):
                self._container.mux(packet)

        if self._container:
            self._container.close()
