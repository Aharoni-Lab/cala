from typing import cast

import cv2
import numpy as np
import pytest
import xarray as xr

from cala.config import Frame, StreamingConfig
from cala.streaming.composer import Runner


@pytest.fixture
def preprocess_config() -> StreamingConfig:
    return cast(
        StreamingConfig,
        {
            "preprocess": {
                "downsample": {
                    "transformer": "Downsampler",
                    "params": {
                        "method": "mean",
                        "dimensions": ["width", "height"],
                        "strides": [2, 2],
                    },
                },
                "denoise": {
                    "transformer": "Denoiser",
                    "params": {
                        "method": "gaussian",
                        "kwargs": {"ksize": (3, 3), "sigmaX": 1.5},
                    },
                    "requires": ["downsample"],
                },
                "glow_removal": {
                    "transformer": "GlowRemover",
                    "params": {},
                    "requires": ["denoise"],
                },
                "background_removal": {
                    "transformer": "BackgroundEraser",
                    "params": {"method": "uniform", "kernel_size": 3},
                    "requires": ["glow_removal"],
                },
                "motion_stabilization": {
                    "transformer": "RigidStabilizer",
                    "params": {"drift_speed": 1, "anchor_frame_index": 0},
                    "requires": ["background_removal"],
                },
            }
        },
    )


@pytest.fixture
def initialization_config() -> StreamingConfig:
    return cast(
        StreamingConfig,
        {
            "initialization": {
                "footprints": {
                    "transformer": "FootprintsInitializer",
                    "params": {
                        "threshold_factor": 0.2,
                        "kernel_size": 3,
                        "distance_metric": cv2.DIST_L2,
                        "distance_mask_size": 5,
                    },
                },
                "traces": {
                    "transformer": "TracesInitializer",
                    "params": {},
                    "n_frames": 3,
                    "requires": ["footprints"],
                },
            }
        },
    )


def test_cyclic_dependency_detection(stabilized_video: xr.DataArray) -> None:
    cyclic_config: StreamingConfig = cast(
        StreamingConfig,
        {
            "initialization": {
                "step1": {
                    "transformer": "FootprintsInitializer",
                    "params": {},
                    "requires": ["step2"],
                },
                "step2": {
                    "transformer": "TracesInitializer",
                    "params": {},
                    "requires": ["step1"],
                },
            }
        },
    )
    runner = Runner(cyclic_config)
    video = stabilized_video
    with pytest.raises(ValueError):
        for idx, frame in enumerate(video):
            frame = Frame(frame, idx)
            while not runner.is_initialized:
                runner.initialize(frame)


def test_preprocess_initialization(preprocess_config: StreamingConfig) -> None:
    runner = Runner(preprocess_config)
    assert runner.config == preprocess_config


def test_preprocess_execution(
    preprocess_config: StreamingConfig, stabilized_video: xr.DataArray
) -> None:
    runner = Runner(preprocess_config)
    video = stabilized_video
    idx, frame = next(iter(enumerate(video)))
    frame = Frame(frame, idx)
    original_shape = frame.array.shape

    # Test preprocessing pipeline
    result = runner.preprocess(frame)

    assert isinstance(result, Frame)

    # Verify dimensions are reduced by downsampling
    processed_frame = result
    assert processed_frame.array.shape[0] == original_shape[0] // 2
    assert processed_frame.array.shape[1] == original_shape[1] // 2


def test_preprocess_dependency_resolution(preprocess_config: StreamingConfig) -> None:
    runner = Runner(preprocess_config)
    execution_order = runner._create_dependency_graph(preprocess_config["preprocess"])

    # Verify correct execution order
    expected_order = [
        "downsample",
        "denoise",
        "glow_removal",
        "background_removal",
        "motion_stabilization",
    ]
    assert list(execution_order) == expected_order


def test_initializer_initialization(initialization_config: StreamingConfig) -> None:
    runner = Runner(initialization_config)
    assert runner.config == initialization_config


def test_initialize_execution(
    initialization_config: StreamingConfig, stabilized_video: xr.DataArray
) -> None:
    runner = Runner(initialization_config)
    video = stabilized_video

    for idx, frame in enumerate(video):
        frame = Frame(frame, idx)
        while not runner.is_initialized:
            runner.initialize(frame)

    assert runner.is_initialized

    assert np.array_equal(
        runner._state.footprintstore.warehouse.coords["id_"].values,
        runner._state.tracestore.warehouse.coords["id_"].values,
    )
    assert np.array_equal(
        runner._state.footprintstore.warehouse.coords["type_"].values,
        runner._state.tracestore.warehouse.coords["type_"].values,
    )
