import logging
from typing import cast

import cv2
import matplotlib.pyplot as plt
import pytest

from cala.log import setup_logger
from cala.streaming.composer import StreamingConfig, Runner, Frame
from cala.streaming.init.common import FootprintsInitializer, TracesInitializer
from cala.streaming.init.odl import (
    PixelStatsInitializer,
    ComponentStatsInitializer,
    ResidualInitializer,
    OverlapsInitializer,
)
from cala.streaming.iterate import (
    TracesUpdater,
    ComponentStatsUpdater,
    PixelStatsUpdater,
    FootprintsUpdater,
)
from cala.streaming.iterate.detect import Detector
from cala.streaming.iterate.overlaps import OverlapsUpdater
from cala.streaming.preprocess import (
    Downsampler,
    Denoiser,
    GlowRemover,
    BackgroundEraser,
    RigidStabilizer,
)

setup_logger("logs/", name="")
logger = logging.getLogger(__name__)


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


def test_preprocess(preprocess_config, raw_calcium_video):
    runner = Runner(preprocess_config)
    video = raw_calcium_video
    for idx, frame in enumerate(video):
        frame = Frame(frame, idx)
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
                    "params": {},
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
        },
    )


def test_initialize(initialization_config, stabilized_video):
    runner = Runner(initialization_config)
    video = stabilized_video

    for idx, frame in enumerate(video):
        frame = Frame(frame, idx)
        runner.initialize(frame)

    assert runner.is_initialized


@pytest.fixture
def integration_config() -> StreamingConfig:
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
                    "params": {},
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
                "traces": {
                    "transformer": TracesUpdater,
                    "params": {"tolerance": 1e-3},
                },
                "pixel_stats": {
                    "transformer": PixelStatsUpdater,
                    "params": {},
                    "requires": ["traces"],
                },
                "component_stats": {
                    "transformer": ComponentStatsUpdater,
                    "params": {},
                    "requires": ["traces"],
                },
                "detect": {
                    "transformer": Detector,
                    "params": {
                        "num_nmf_residual_frames": 10,
                        "gaussian_std": 4.0,
                        "spatial_threshold": 0.8,
                        "temporal_threshold": 0.8,
                    },
                    "requires": ["pixel_stats", "component_stats"],
                },
                "footprints": {
                    "transformer": FootprintsUpdater,
                    "params": {"boundary_expansion_pixels": 1},
                    "requires": ["detect"],
                },
                "overlaps": {
                    "transformer": OverlapsUpdater,
                    "params": {},
                    "requires": ["footprints"],
                },
            },
        },
    )


@pytest.mark.timeout(30)
def test_iteration(integration_config, simply_denoised):
    runner = Runner(integration_config)
    video = simply_denoised

    for idx, frame in enumerate(video):
        frame = Frame(frame, idx)
        plt.imsave(f"frame{idx}.png", frame.array)

        if not runner.is_initialized:
            runner.initialize(frame)
            continue

        logger.info(f"Frame: {idx}")
        runner.iterate(frame)


@pytest.mark.timeout(30)
def test_integration(integration_config, raw_calcium_video):
    runner = Runner(integration_config)
    video = raw_calcium_video

    for idx, frame in enumerate(video):
        frame = Frame(frame, idx)
        frame = runner.preprocess(frame)
        plt.imsave(f"frame{idx}.png", frame.array)

        if not runner.is_initialized:
            runner.initialize(frame)
            continue

        runner.iterate(frame)
