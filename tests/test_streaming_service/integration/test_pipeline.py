from typing import cast

import cv2
import pytest

from cala.streaming.composer import StreamingConfig, Runner
from cala.streaming.init.common import FootprintsInitializer, TracesInitializer
from cala.streaming.init.odl import (
    PixelStatsInitializer,
    ComponentStatsInitializer,
    ResidualInitializer,
)
from cala.streaming.init.odl.overlaps import OverlapsInitializer
from cala.streaming.iterate.traces import TracesUpdater
from cala.streaming.preprocess import (
    Downsampler,
    Denoiser,
    GlowRemover,
    BackgroundEraser,
    RigidStabilizer,
)


@pytest.fixture
def preprocess_config() -> StreamingConfig:
    return cast(
        StreamingConfig,
        {
            "preprocess": {
                "downsample": {
                    "transformer": Downsampler,
                    "params": {
                        "method": "mean",
                        "dimensions": ["width", "height"],
                        "strides": [2, 2],
                    },
                },
                "denoise": {
                    "transformer": Denoiser,
                    "params": {
                        "method": "gaussian",
                        "kwargs": {"ksize": (3, 3), "sigmaX": 1.5},
                    },
                    "requires": ["downsample"],
                },
                "glow_removal": {
                    "transformer": GlowRemover,
                    "params": {},
                    "requires": ["denoise"],
                },
                "background_removal": {
                    "transformer": BackgroundEraser,
                    "params": {"method": "uniform", "kernel_size": 3},
                    "requires": ["glow_removal"],
                },
                "motion_stabilization": {
                    "transformer": RigidStabilizer,
                    "params": {"drift_speed": 1, "anchor_frame_index": 0},
                    "requires": ["background_removal"],
                },
            }
        },
    )


def test_preprocess_execution(preprocess_config, raw_calcium_video):
    runner = Runner(preprocess_config)
    video, _, _ = raw_calcium_video
    for frame in video:
        frame = runner.preprocess(frame)


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
                    "requires": ["traces"],
                },
                "residual": {
                    "transformer": ResidualInitializer,
                    "params": {"buffer_length": 3},
                    "n_frames": 3,
                    "requires": ["footprints", "traces"],
                },
                "overlap_groups": {
                    "transformer": OverlapsInitializer,
                    "params": {},
                    "requires": ["footprints"],
                },
            },
            "iteration": {
                "update_trace": {
                    "transformer": TracesUpdater,
                    "params": {"tolerance": 1e-3},
                }
            },
        },
    )


def test_initialize_execution(initialization_config, stabilized_video):
    runner = Runner(initialization_config)
    video, _, _ = stabilized_video

    for frame in video:
        runner.initialize(frame)

    assert runner.is_initialized


@pytest.fixture
def streaming_config() -> StreamingConfig:
    return cast(
        StreamingConfig,
        {
            "preprocess": {
                "downsample": {
                    "transformer": Downsampler,
                    "params": {
                        "method": "mean",
                        "dimensions": ["width", "height"],
                        "strides": [2, 2],
                    },
                },
                "denoise": {
                    "transformer": Denoiser,
                    "params": {
                        "method": "gaussian",
                        "kwargs": {"ksize": (3, 3), "sigmaX": 1.5},
                    },
                    "requires": ["downsample"],
                },
                "glow_removal": {
                    "transformer": GlowRemover,
                    "params": {"learning_rate": 0.1},
                    "requires": ["denoise"],
                },
                # "background_removal": {
                #     "transformer": BackgroundEraser,
                #     "params": {"method": "uniform", "kernel_size": 3},
                #     "requires": ["glow_removal"],
                # },
                "motion_stabilization": {
                    "transformer": RigidStabilizer,
                    "params": {"drift_speed": 1, "anchor_frame_index": 0},
                    "requires": ["glow_removal"],
                },
            },
            "initialization": {
                "footprints": {
                    "transformer": FootprintsInitializer,
                    "params": {
                        "threshold_factor": 0.5,
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
                    "requires": ["traces"],
                },
                "residual": {
                    "transformer": ResidualInitializer,
                    "params": {"buffer_length": 3},
                    "n_frames": 3,
                    "requires": ["footprints", "traces"],
                },
                "overlap_groups": {
                    "transformer": OverlapsInitializer,
                    "params": {},
                    "requires": ["footprints"],
                },
            },
            "iteration": {
                "update_trace": {
                    "transformer": TracesUpdater,
                    "params": {"tolerance": 1e-3},
                }
            },
        },
    )


def test_streaming_execution(streaming_config, raw_calcium_video):
    # import matplotlib.pyplot as plt

    runner = Runner(streaming_config)
    video, _, _ = raw_calcium_video

    for idx, frame in enumerate(video):
        # plt.imsave(f"preprocess_{idx}.png", frame)
        frame = runner.preprocess(frame)
        # plt.imsave(f"postprocess_{idx}.png", frame)

        if not runner.is_initialized:
            runner.initialize(frame)
            continue

        runner.iterate(frame)
