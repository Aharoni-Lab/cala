import os
import shutil
import time
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

os.environ["CALA_CONFIG_PATH"] = "tests/test_gui/test_config.yaml"
os.environ["NODE_ENV"] = "development"
os.environ["PYTHONASYNCIODEBUG"] = "1"

from fastapi.testclient import TestClient

from cala.config import Config
from cala.gui.app import get_app
from cala.gui.nodes import FrameStreamer
from cala.gui.nodes.frame_streamer import FrameStreamerParams
from cala.core.execute import Executor


class TestFrameStreamer:
    def frame(self, dims) -> xr.DataArray:
        """Generate a single frame"""
        t = time.time()
        frame = xr.DataArray(
            np.ones(dims, dtype=np.uint8) * int(255 * (np.sin(t) + 1) / 2), dims=("height", "width")
        )
        return frame

    def test_frame_streamer(self):
        dims = (100, 100)
        fps = 30
        segment_duration_in_seconds = 2
        stream_dir = (Path(".") / "output" / "test_stream").resolve()
        segment_filename = "stream%d.ts"

        params = FrameStreamerParams(
            frame_rate=fps,
            stream_dir=stream_dir,
            segment_filename=segment_filename,
            segment_duration_in_seconds=segment_duration_in_seconds,
        )
        frame_streamer = FrameStreamer(params)

        segment_count = 100
        # EXT-T-TARGETDURATION is a second longer than the specified segment_duration_in_seconds
        # https://datatracker.ietf.org/doc/html/draft-pantos-http-live-streaming-19#section-4.3.3.1
        for _ in range(fps * (segment_duration_in_seconds + 1) * segment_count):
            fr = self.frame(dims)
            frame_streamer.learn_one(fr)
            frame_streamer.transform_one(fr)
            # time.sleep(1 / fps)

        assert stream_dir.is_dir()
        assert len(list(stream_dir.iterdir())) > 0
        assert Path(frame_streamer.hls_manifest).resolve().exists()
        assert stream_dir.glob(segment_filename.replace("%d", "*"))

        shutil.rmtree(stream_dir)


class TestGUIStream:
    @pytest.fixture(scope="class")
    def client(self):
        return TestClient(get_app())

    def test_gui_spinup(self, client):
        response = client.get("/")
        assert response.status_code == 200

    def test_gui_static(self, client):
        response = client.get("/dist/main.js")
        assert response.status_code == 200

    @pytest.fixture(scope="class")
    def config(self):
        root_path = Path(__file__).parents[2]
        config = Config.from_yaml(root_path / "tests/test_gui/test_config.yaml")
        return config

    @pytest.fixture(scope="class")
    def runner(self, config):
        runner = Executor(config)
        return runner

    def test_runner_preprocess(self, config, runner, raw_calcium_video):
        for _idx, frame in enumerate(raw_calcium_video):
            runner.preprocess(frame)
            time.sleep(0.03)

        shutil.rmtree(config.output_dir)

    def test_stream_read(self, runner, client, config, raw_calcium_video):
        video = xr.concat([raw_calcium_video] * 5, dim="frame")
        for _idx, frame in enumerate(video):
            runner.preprocess(frame)
            time.sleep(0.03 / 100)

        response = client.get("/prep_movie/stream.m3u8")
        assert response.status_code == 200

        shutil.rmtree(config.output_dir)
