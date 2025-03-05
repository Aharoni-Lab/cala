from typing import cast

import numpy as np
import pytest
import xarray as xr
from river.base import Transformer

from cala.streaming.runner import (
    Runner,
    Frame,
    NeuronFootprints,
    NeuronTraces,
    TransformerMeta,
    Config,
)
from tests.conftest import stabilized_video


# Mock transformers for testing
class MockMotionCorrection(Transformer, metaclass=TransformerMeta):
    def __init__(self, max_shift: int = 10):
        self.max_shift = max_shift
        self.frame = None

    def learn_one(self, frame: Frame) -> None:
        self.frame = frame
        return None

    def transform_one(self, _=None) -> Frame:
        # Simulate motion correction by returning the same frame
        return Frame(self.frame)


class MockNeuronDetection(Transformer, metaclass=TransformerMeta):
    def __init__(self, num_components: int = 100):
        self.num_components = num_components
        self.frame = None

    def learn_one(self, frame: Frame) -> None:
        self.frame = frame
        return None

    def transform_one(self, _=None) -> NeuronFootprints:
        # Create mock neuron footprints
        data = np.random.rand(self.num_components, *self.frame.shape)
        return NeuronFootprints(
            data,
            dims=["neuron", "height", "width"],
            coords={
                "neuron": range(self.num_components),
                "height": self.frame.coords["height"],
                "width": self.frame.coords["width"],
            },
        )


class MockTraceExtractor(Transformer, metaclass=TransformerMeta):
    def __init__(self, method: str = "pca"):
        self.method = method

    def learn_one(self, frame: Frame) -> None:
        return None

    def transform_one(self, neuron_footprints: NeuronFootprints) -> NeuronTraces:
        # Create mock traces
        data = np.random.rand(len(neuron_footprints), 100)  # 100 timepoints
        return NeuronTraces(
            data,
            dims=["neuron", "time"],
            coords={"neuron": neuron_footprints.coords["neuron"], "time": range(100)},
        )


@pytest.fixture
def basic_config() -> Config:
    return cast(
        Config,
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


def test_runner_initialization(basic_config):
    runner = Runner(basic_config)
    assert runner.config == basic_config


def test_runner_dependency_resolution(basic_config, stabilized_video):
    runner = Runner(basic_config)
    video, _, _ = stabilized_video
    for frame in video:
        while not runner.is_initialized:
            state = runner.initialize(frame=frame)

    assert state is not None


def test_cyclic_dependency_detection(stabilized_video):
    cyclic_config: Config = cast(
        Config,
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
    video, _, _ = stabilized_video
    with pytest.raises(ValueError, match="Transformer dependencies contain cycles"):
        for frame in video:
            while not runner.is_initialized:
                runner.initialize(frame)


def test_state_updates(basic_config, stabilized_video):
    runner = Runner(basic_config)
    video, _, _ = stabilized_video
    for frame in video:
        while not runner.is_initialized:
            state = runner.initialize(frame)
    # Check if state contains expected attributes
    assert hasattr(state, "neuron_footprints")
    assert hasattr(state, "neuron_traces")
    assert isinstance(state.neuron_footprints, xr.DataArray)
    assert isinstance(state.neuron_traces, xr.DataArray)


def test_transformer_type_injection(basic_config, stabilized_video):
    runner = Runner(basic_config)
    video, _, _ = stabilized_video
    for frame in video:
        while not runner.is_initialized:
            state = runner.initialize(frame)
    # Verify the shapes and dimensions of the outputs
    assert state.neuron_footprints.dims == ("neuron", "height", "width")
    assert state.neuron_traces.dims == ("neuron", "time")
    assert len(state.neuron_footprints.neuron) == 10  # As specified in config


# def test_with_footprint_and_traces
