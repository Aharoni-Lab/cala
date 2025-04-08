import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import xarray as xr

from cala.config import Config
from cala.log import setup_logger
from cala.streaming.composer import Runner
from cala.streaming.util import package_frame

setup_logger(Path(__file__).parent / "logs", name="")
logger = logging.getLogger(__name__)


def load_config(config_name: str) -> Config:
    config_path = Path(__file__).parent / f"{config_name}.yaml"
    return Config.from_yaml(config_path)


def test_preprocess(raw_calcium_video: xr.DataArray) -> None:
    config = load_config("integration")
    runner = Runner(config.pipeline)
    video = raw_calcium_video
    for idx, frame in enumerate(video):
        frame = package_frame(frame.values, idx)
        runner.preprocess(frame)


def test_initialize(stabilized_video: xr.DataArray) -> None:
    config = load_config("integration")
    runner = Runner(config.pipeline)
    video = stabilized_video

    for idx, frame in enumerate(video):
        frame = package_frame(frame.values, idx)
        while not runner.is_initialized:
            runner.initialize(frame)

    assert runner.is_initialized


@pytest.mark.parametrize("video", ["raw_calcium_video", "simply_denoised"])
@pytest.mark.timeout(30)
def test_integration(video: str, request) -> None:
    video = request.getfixturevalue(video)
    config = load_config("integration")
    runner = Runner(config.pipeline)

    for idx, frame in enumerate(video):
        frame = package_frame(frame.values, idx)
        frame = runner.preprocess(frame)
        plt.imsave(f"frame{idx}.png", frame.array)

        if not runner.is_initialized:
            runner.initialize(frame)
            continue

        runner.iterate(frame)
