from dataclasses import dataclass, field
from typing import cast
from uuid import uuid4

import cv2
import numpy as np
import pytest
import xarray as xr
from river.base import Transformer

from cala.streaming.composer import StreamingConfig, Runner, Frame
from cala.streaming.core import Parameters, Component
from cala.streaming.init.common import FootprintsInitializer, TracesInitializer
from cala.streaming.preprocess import (
    RigidStabilizer,
    BackgroundEraser,
    Denoiser,
    Downsampler,
    GlowRemover,
)
from cala.streaming.stores.common import Footprints, Traces
from tests.conftest import stabilized_video


@dataclass
class MockMotionCorrectionParams(Parameters):
    """Parameters for mock motion correction."""

    max_shift: int = 10
    """Maximum allowed shift in pixels."""

    def validate(self) -> None:
        if self.max_shift < 0:
            raise ValueError("max_shift must be non-negative")


# Mock transformers for testing
@dataclass
class MockMotionCorrection(Transformer):
    params: MockMotionCorrectionParams = field(
        default_factory=MockMotionCorrectionParams
    )
    frame_: xr.DataArray = field(init=False)

    def learn_one(self, frame: xr.DataArray) -> None:
        self.frame_ = frame
        return None

    def transform_one(self, _=None) -> xr.DataArray:
        # Simulate motion correction by returning the same frame
        if self.frame_ is None:
            raise ValueError("No frame has been learned yet")
        return xr.DataArray(self.frame_)


@dataclass
class MockNeuronDetectionParams(Parameters):
    """Parameters for mock neuron detection."""

    num_components: int = 100
    """Number of neural components to detect."""

    def validate(self) -> None:
        if self.num_components <= 0:
            raise ValueError("num_components must be positive")


@dataclass
class MockNeuronDetection(Transformer):
    params: MockNeuronDetectionParams = field(default_factory=MockNeuronDetectionParams)
    frame_: xr.DataArray = field(init=False)

    def learn_one(self, frame: xr.DataArray) -> None:
        self.frame_ = frame
        return None

    def transform_one(self, _=None) -> Footprints:
        # Create mock neuron footprints
        if self.frame_ is None:
            raise ValueError("No frame has been learned yet")
        data = np.random.rand(self.params.num_components, *self.frame_.shape)
        return xr.DataArray(
            data,
            dims=["components", "height", "width"],
            coords={
                "type_": (
                    ["components"],
                    [Component.NEURON] * (self.params.num_components - 1)
                    + [Component.BACKGROUND],
                ),
                "id_": (
                    ["components"],
                    [uuid4() for _ in range(self.params.num_components)],
                ),
            },
        )


@dataclass
class MockTraceExtractorParams(Parameters):
    """Parameters for mock trace extraction."""

    method: str = "pca"
    """Method used for trace extraction."""

    def validate(self) -> None:
        if self.method not in ["pca", "nmf"]:
            raise ValueError("method must be one of ['pca', 'nmf']")


@dataclass
class MockTraceExtractor(Transformer):
    params: MockTraceExtractorParams = field(default_factory=MockTraceExtractorParams)
    frame_: xr.DataArray = field(init=False)

    def learn_one(self, frame: xr.DataArray) -> None:
        self.frame_ = frame
        return None

    def transform_one(self, neuron_footprints: Footprints) -> Traces:
        if self.frame_ is None:
            raise ValueError("No frame has been learned yet")
        # Create mock traces
        data = np.random.rand(len(neuron_footprints), 100)  # 100 timepoints
        return xr.DataArray(
            data,
            dims=["components", "frames"],
            coords={
                "type_": (
                    ["components"],
                    neuron_footprints.coords["type_"].values,
                ),
                "id_": (
                    ["components"],
                    neuron_footprints.coords["id_"].values,
                ),
            },
        )


@pytest.fixture
def basic_config() -> StreamingConfig:
    return cast(
        StreamingConfig,
        {
            "initialization": {
                "motion_correction": {
                    "transformer": MockMotionCorrection,
                    "params": {"max_shift": 10},
                },
                "neuron_detection": {
                    "transformer": MockNeuronDetection,
                    "params": {"num_components": 10},
                    "requires": ["motion_correction"],
                },
                "trace_extraction": {
                    "transformer": MockTraceExtractor,
                    "params": {"method": "pca"},
                    "requires": ["neuron_detection"],
                },
            }
        },
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
            }
        },
    )


def test_runner_initialization(basic_config):
    runner = Runner(basic_config)
    assert runner.config == basic_config


def test_runner_dependency_resolution(basic_config, stabilized_video):
    runner = Runner(basic_config)
    video = stabilized_video
    for idx, frame in enumerate(video):
        frame = Frame(frame, idx)
        while not runner.is_initialized:
            runner.initialize(frame=frame)

    assert runner._state.footprintstore.warehouse.sizes == {
        "components": 10,
        "width": 512,
        "height": 512,
    }
    assert runner._state.tracestore.warehouse.sizes == {
        "components": 10,
        "frames": 100,
    }
    assert np.array_equal(
        runner._state.footprintstore.warehouse.coords["id_"].values,
        runner._state.tracestore.warehouse.coords["id_"].values,
    )
    assert np.array_equal(
        runner._state.footprintstore.warehouse.coords["type_"].values,
        runner._state.tracestore.warehouse.coords["type_"].values,
    )


def test_cyclic_dependency_detection(stabilized_video):
    cyclic_config: StreamingConfig = cast(
        StreamingConfig,
        {
            "initialization": {
                "step1": {
                    "transformer": MockMotionCorrection,
                    "params": {},
                    "requires": ["step2"],
                },
                "step2": {
                    "transformer": MockNeuronDetection,
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


def test_state_updates(basic_config, stabilized_video):
    runner = Runner(basic_config)
    video = stabilized_video
    for idx, frame in enumerate(video):
        frame = Frame(frame, idx)
        while not runner.is_initialized:
            runner.initialize(frame)
    # Check if state contains expected attributes

    assert runner._state.footprintstore.warehouse.sizes != 0
    assert runner._state.tracestore.warehouse.sizes != 0


def test_preprocess_initialization(preprocess_config):
    runner = Runner(preprocess_config)
    assert runner.config == preprocess_config


def test_preprocess_execution(preprocess_config, stabilized_video):
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


def test_preprocess_dependency_resolution(preprocess_config):
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


def test_initializer_initialization(initialization_config):
    runner = Runner(initialization_config)
    assert runner.config == initialization_config


def test_initialize_execution(initialization_config, stabilized_video):
    runner = Runner(initialization_config)
    video = stabilized_video

    for idx, frame in enumerate(video):
        frame = Frame(frame, idx)
        while not runner.is_initialized:
            runner.initialize(frame)

    assert runner.is_initialized
