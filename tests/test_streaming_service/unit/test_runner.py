from typing import cast

import numpy as np
import pytest
import xarray as xr
from river.base import Transformer

from cala.streaming.initialization.meta import TransformerMeta
from cala.streaming.pipe_config import StreamingConfig
from cala.streaming.runner import Runner
from cala.streaming.types import NeuronFootprints, NeuronTraces
from tests.conftest import stabilized_video


# Mock transformers for testing
class MockMotionCorrection(Transformer, metaclass=TransformerMeta):
    def __init__(self, max_shift: int = 10):
        self.max_shift = max_shift
        self.frame = None

    def learn_one(self, frame: xr.DataArray) -> None:
        self.frame = frame
        return None

    def transform_one(self, _=None) -> xr.DataArray:
        # Simulate motion correction by returning the same frame
        return xr.DataArray(self.frame)


class MockNeuronDetection(Transformer, metaclass=TransformerMeta):
    def __init__(self, num_components: int = 100):
        self.num_components = num_components
        self.frame = None

    def learn_one(self, frame: xr.DataArray) -> None:
        self.frame = frame
        return None

    def transform_one(self, _=None) -> NeuronFootprints:
        # Create mock neuron footprints
        data = np.random.rand(self.num_components, *self.frame.shape)
        return NeuronFootprints(
            data,
            dims=["components", "height", "width"],
            coords={
                "components": range(self.num_components),
                "height": self.frame.coords["height"],
                "width": self.frame.coords["width"],
            },
        )


class MockTraceExtractor(Transformer, metaclass=TransformerMeta):
    def __init__(self, method: str = "pca"):
        self.method = method

    def learn_one(self, frame: xr.DataArray) -> None:
        return None

    def transform_one(self, neuron_footprints: NeuronFootprints) -> NeuronTraces:
        # Create mock traces
        data = np.random.rand(len(neuron_footprints), 100)  # 100 timepoints
        return NeuronTraces(
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


def test_runner_initialization(basic_config):
    runner = Runner(basic_config)
    assert runner.config == basic_config


def test_runner_dependency_resolution(basic_config, stabilized_video):
    runner = Runner(basic_config)
    video, _, _ = stabilized_video
    for frame in video:
        while not runner.is_initialized:
            state = runner.initialize(frame=frame)

    assert state.footprints.warehouse.sizes == {
        "components": 10,
        "width": 512,
        "height": 512,
    }
    assert state.traces.warehouse.sizes == {"components": 10, "frames": 100}
    assert np.array_equal(
        state.footprints.warehouse.coords["id_"].values,
        state.traces.warehouse.coords["id_"].values,
    )
    assert np.array_equal(
        state.footprints.warehouse.coords["type_"].values,
        state.traces.warehouse.coords["type_"].values,
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
    neuron_footprint = state.get_observable_x_component(NeuronFootprints)
    neuron_traces = state.get_observable_x_component(NeuronTraces)
    assert neuron_footprint.__len__() != 0
    assert neuron_traces.__len__() != 0
    assert neuron_traces.__len__() == neuron_footprint.__len__()


# def test_with_footprint_and_traces
