from collections.abc import Sequence
from typing import Any

from cala.streaming.composer.pipe_config import (
    InitializationStep,
    IterationStep,
    PreprocessStep,
    StreamingConfig,
)


class MockTransformer:
    def __init__(self, **kwargs: Any) -> None:
        self.params = kwargs


def test_preprocess_step_valid() -> None:
    # Test basic configuration
    step: PreprocessStep = {
        "transformer": MockTransformer,
        "params": {"param1": "value1"},
    }
    assert step["transformer"] == MockTransformer
    assert step["params"] == {"param1": "value1"}

    # Test with optional requires field
    step_with_requires: PreprocessStep = {
        "transformer": MockTransformer,
        "params": {"param1": "value1"},
        "requires": ["dep1", "dep2"],
    }
    assert isinstance(step_with_requires["requires"], Sequence)
    assert step_with_requires["requires"] == ["dep1", "dep2"]


def test_initialization_step_valid() -> None:
    # Test basic configuration
    step: InitializationStep = {
        "transformer": MockTransformer,
        "params": {"param1": "value1"},
    }
    assert step["transformer"] == MockTransformer
    assert step["params"] == {"param1": "value1"}

    # Test with optional requires field
    step_with_requires: InitializationStep = {
        "transformer": MockTransformer,
        "params": {"param1": "value1"},
        "requires": ["dep1", "dep2"],
    }
    assert isinstance(step_with_requires["requires"], Sequence)
    assert step_with_requires["requires"] == ["dep1", "dep2"]


def test_extraction_step_valid() -> None:
    # Test basic configuration
    step: IterationStep = {
        "transformer": MockTransformer,
        "params": {"param1": "value1"},
    }
    assert step["transformer"] == MockTransformer
    assert step["params"] == {"param1": "value1"}

    # Test with optional requires field
    step_with_requires: IterationStep = {
        "transformer": MockTransformer,
        "params": {"param1": "value1"},
        "requires": ["dep1", "dep2"],
    }
    assert isinstance(step_with_requires["requires"], Sequence)
    assert step_with_requires["requires"] == ["dep1", "dep2"]


def test_streaming_config_valid() -> None:
    # Test minimal configuration
    config: StreamingConfig = {
        "preprocess": {},
        "initialization": {},
        "iteration": {},
    }
    assert isinstance(config["preprocess"], dict)
    assert isinstance(config["initialization"], dict)
    assert isinstance(config["iteration"], dict)

    # Test full configuration with all optional fields
    full_config: StreamingConfig = {
        "preprocess": {
            "step1": {
                "transformer": MockTransformer,
                "params": {"param1": "value1"},
            }
        },
        "initialization": {
            "step2": {
                "transformer": MockTransformer,
                "params": {"param2": "value2"},
                "requires": ["step1"],
            }
        },
        "iteration": {
            "step3": {
                "transformer": MockTransformer,
                "params": {"param3": "value3"},
                "requires": ["step2"],
            }
        },
    }

    assert "step1" in full_config["preprocess"]
    assert "step2" in full_config["initialization"]
    assert "step3" in full_config["iteration"]


def test_streaming_config_dependencies() -> None:
    # Test configuration with dependencies between steps
    config: StreamingConfig = {
        "preprocess": {},
        "initialization": {
            "motion_correction": {
                "transformer": MockTransformer,
                "params": {"max_shift": 10},
            },
            "neuron_detection": {
                "transformer": MockTransformer,
                "params": {"num_components": 10},
                "requires": ["motion_correction"],
            },
            "trace_extraction": {
                "transformer": MockTransformer,
                "params": {"method": "pca"},
                "requires": ["neuron_detection"],
            },
        },
        "iteration": {},
    }

    # Verify the dependency chain
    neuron_detection = config["initialization"]["neuron_detection"]
    trace_extraction = config["initialization"]["trace_extraction"]

    assert "requires" in neuron_detection
    assert "requires" in trace_extraction
    assert "motion_correction" in neuron_detection["requires"]
    assert "neuron_detection" in trace_extraction["requires"]
