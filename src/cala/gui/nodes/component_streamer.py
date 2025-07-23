import os
from dataclasses import dataclass
from pathlib import Path

import av
import xarray as xr
from river.base import Transformer

from cala.models import Params
from cala.stores.common import Footprints


@dataclass
class ComponentStreamerParams(Params):
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
class ComponentStreamer(Transformer):
    params: ComponentStreamerParams
    footprint_projection_: xr.DataArray | None = None
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

    def learn_one(self, footprints: Footprints) -> None:
        self.footprint_projection_ = footprints.max(dim=self.params.component_dim)

    def transform_one(self, frame: xr.DataArray) -> xr.DataArray:
        self.stream.width = self.footprint_projection_.sizes["width"]
        self.stream.height = self.footprint_projection_.sizes["height"]

        try:
            # Create frame
            frame_sample = av.VideoFrame.from_ndarray(
                self.footprint_projection_.to_numpy(), format="gray"
            ).reformat(format="yuv420p")

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
