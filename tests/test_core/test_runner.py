import cv2
import numpy as np
import pytest
import xarray as xr

from cala.core.execute import Executor
from cala.models.axis import AXIS
from cala.models.spec import (
    NodeSpec,
    PipeSpec,
)
from cala.util.new import package_frame


@pytest.fixture
def preprocess_config() -> PipeSpec:
    return PipeSpec(
        buff={"buffer_size": 100},
        prep={
            "downsample": NodeSpec(
                id="Downsampler",
                params={
                    "method": "mean",
                    "dimensions": ["width", "height"],
                    "strides": [2, 2],
                },
            ),
            "denoise": NodeSpec(
                id="Denoiser",
                params={
                    "method": "gaussian",
                    "kwargs": {"ksize": (3, 3), "sigmaX": 1.5},
                },
                requires=["downsample"],
            ),
            "glow_removal": NodeSpec(
                id="GlowRemover",
                params={},
                requires=["denoise"],
            ),
            "background_removal": NodeSpec(
                id="BackgroundEraser",
                params={"method": "uniform", "kernel_size": 3},
                requires=["glow_removal"],
            ),
            "motion_stabilization": NodeSpec(
                id="RigidStabilizer",
                params={"drift_speed": 1, "anchor_frame_index": 0},
                requires=["background_removal"],
            ),
        },
        init={},
        iter={},
    )


@pytest.fixture
def initialization_config() -> PipeSpec:
    return PipeSpec(
        buff={"buffer_size": 100},
        prep={},
        init={
            "footprints": NodeSpec(
                id="FootprintsInitializer",
                params={
                    "threshold_factor": 0.2,
                    "kernel_size": 3,
                    "distance_metric": cv2.DIST_L2,
                    "distance_mask_size": 5,
                },
            ),
            "traces": NodeSpec(
                id="TracesInitializer",
                params={},
                n_frames=3,
                requires=["footprints"],
            ),
        },
        iter={},
    )


def test_cyclic_dependency_detection(stabilized_video: xr.DataArray, tmp_path) -> None:
    cyclic_config = PipeSpec(
        buff={"buffer_size": 100},
        prep={},
        init={
            "step1": NodeSpec(
                id="FootprintsInitializer",
                params={},
                requires=["step2"],
            ),
            "step2": NodeSpec(
                id="TracesInitializer",
                params={},
                requires=["step1"],
            ),
        },
        iter={},
    )
    runner = Executor(cyclic_config, tmp_path)
    video = stabilized_video
    with pytest.raises(ValueError):
        for idx, frame in enumerate(video):
            frame = package_frame(frame.values, idx)
            while not runner.is_initialized:
                runner.initialize(frame)


def test_preprocess_initialization(preprocess_config: PipeSpec, tmp_path) -> None:
    runner = Executor(preprocess_config, tmp_path)
    assert runner.pipeline == preprocess_config


def test_preprocess_execution(
    preprocess_config: PipeSpec, stabilized_video: xr.DataArray, tmp_path
) -> None:
    runner = Executor(preprocess_config, tmp_path)
    video = stabilized_video
    idx, frame = next(iter(enumerate(video)))
    frame = package_frame(frame.values, idx)
    original_shape = frame.shape

    # Test preprocessing pipeline
    result = runner.preprocess(frame)

    assert isinstance(result, xr.DataArray)
    assert AXIS.frame_coord in result.coords
    assert AXIS.timestamp_coord in result.coords

    # Verify dimensions are reduced by downsampling
    assert result.shape[0] == original_shape[0] // 2
    assert result.shape[1] == original_shape[1] // 2


def test_preprocess_dependency_resolution(preprocess_config: PipeSpec, tmp_path) -> None:
    runner = Executor(preprocess_config, tmp_path)
    execution_order = runner._create_dependency_graph(preprocess_config.prep)

    # Verify correct execution order
    expected_order = [
        "downsample",
        "denoise",
        "glow_removal",
        "background_removal",
        "motion_stabilization",
    ]
    assert list(execution_order) == expected_order


def test_initializer_initialization(initialization_config: PipeSpec, tmp_path) -> None:
    runner = Executor(initialization_config, tmp_path)
    assert runner.pipeline == initialization_config


def test_initialize_execution(
    initialization_config: PipeSpec, stabilized_video: xr.DataArray, tmp_path
) -> None:
    runner = Executor(initialization_config, tmp_path)
    video = stabilized_video

    for idx, frame in enumerate(video):
        frame = package_frame(frame.values, idx)
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
