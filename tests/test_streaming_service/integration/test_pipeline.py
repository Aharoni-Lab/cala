from typing import cast

import cv2
import pytest

from cala.streaming.composer import StreamingConfig, Runner
from cala.streaming.init.common import FootprintsInitializer, TracesInitializer
from cala.streaming.init.odl import PixelStatsInitializer
from cala.streaming.init.odl.component_stats import ComponentStatsInitializer


@pytest.fixture
def initialization_config() -> StreamingConfig:
    return cast(
        StreamingConfig,
        {
            "initialization": {
                "footprints": {
                    "transformer": FootprintsInitializer,
                    "params": {
                        "threshold_factor": 0.2,
                        "kernel_size": 3,
                        "distance_metric": cv2.DIST_L2,
                        "distance_mask_size": 5,
                    },
                },
                "traces": {
                    "transformer": TracesInitializer,
                    "params": {"component_axis": "components", "frames_axis": "frame"},
                    "n_frames": 3,
                    "requires": ["footprints"],
                },
                "pixel_stats": {
                    "transformer": PixelStatsInitializer,
                    "params": {},
                    "n_frames": 3,
                    "requires": ["traces"],
                },
                "component_stats": {
                    "transformer": ComponentStatsInitializer,
                    "params": {},
                    "n_frames": 3,
                    "requires": ["pixel_stats"],
                },
            }
        },
    )


def test_initializer_initialization(initialization_config):
    runner = Runner(initialization_config)
    assert runner.config == initialization_config


def test_initialize_execution(initialization_config, stabilized_video):
    runner = Runner(initialization_config)
    video, _, _ = stabilized_video

    # video.values = np.zeros_like(video)
    # video[1] = video[0] + 1
    # video[2] = video[1] + 1
    # video[3] = video[2] + 1

    for frame in video:
        runner.initialize(frame)
        if runner.is_initialized:
            break

    assert runner.is_initialized
