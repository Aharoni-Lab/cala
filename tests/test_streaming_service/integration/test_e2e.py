import logging
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import pytest
import xarray as xr
import yaml

from cala.config import Config
from cala.log import setup_logger
from cala.streaming.composer import Runner
from cala.streaming.util import package_frame

setup_logger(Path(__file__).parent / "logs", name="")
logger = logging.getLogger(__name__)


def load_config(config_name: str) -> Config:
    config_path = Path(__file__).parent / f"{config_name}.yaml"

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        test_file = tmp_path / "test_data.tif"
        test_file.touch()

        # original config
        with open(config_path) as f:
            config_data = yaml.safe_load(f)

        # input_files = temporary file
        config_data["input_files"] = [str(test_file)]

        # temporary config file
        temp_config_path = tmp_path / "temp_config.yaml"
        with open(temp_config_path, "w") as f:
            yaml.dump(config_data, f)

        return Config.from_yaml(str(temp_config_path))


def test_preprocess(raw_calcium_video: xr.DataArray) -> None:
    config = load_config("integration")
    runner = Runner(config.pipeline, config.output_dir)
    video = raw_calcium_video
    for idx, frame in enumerate(video):
        frame = package_frame(frame.values, idx)
        runner.preprocess(frame)


def test_initialize(stabilized_video: xr.DataArray) -> None:
    config = load_config("integration")
    runner = Runner(config.pipeline, config.output_dir)
    video = stabilized_video

    for idx, frame in enumerate(video):
        frame = package_frame(frame.values, idx)
        if not runner.is_initialized:
            runner.initialize(frame)

    assert runner.is_initialized


@pytest.mark.parametrize("video", ["simply_denoised", "raw_calcium_video"])
@pytest.mark.timeout(30)
def test_integration(video: str, request) -> None:
    video = request.getfixturevalue(video)
    config = load_config("integration")
    runner = Runner(config.pipeline, config.output_dir)

    for idx, frame in enumerate(video):
        frame = package_frame(frame.values, idx)
        frame = runner.preprocess(frame)
        plt.imsave(f"frame{idx}.png", frame)

        if not runner.is_initialized:
            runner.initialize(frame)
            continue

        runner.iterate(frame)
