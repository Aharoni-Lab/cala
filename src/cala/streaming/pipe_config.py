from typing import TypedDict, Any, Sequence, NotRequired


class PreprocessStep(TypedDict):
    transformer: type
    params: dict[str, Any]
    requires: NotRequired[Sequence[str]]


class InitializationStep(TypedDict):
    transformer: type  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class ExtractionStep(TypedDict):
    transformer: type  # The transformer class
    params: dict[str, Any]  # Parameters for the transformer
    requires: NotRequired[Sequence[str]]  # Optional dependencies


class StreamingConfig(TypedDict):
    preprocess: dict[str, PreprocessStep]
    initialization: dict[str, InitializationStep]
    extraction: dict[str, ExtractionStep]
    required: NotRequired[Sequence[str]]


# Example config

# config = {
#     "initialization": {
#         "motion_correction": {
#             "transformer": MockMotionCorrection,
#             "params": {"max_shift": 10},
#         },
#         "neuron_detection": {
#             "transformer": MockNeuronDetection,
#             "params": {"num_components": 10},
#             "requires": ["motion_correction"],
#         },
#         "trace_extraction": {
#             "transformer": MockTraceExtractor,
#             "params": {"method": "pca"},
#             "requires": ["neuron_detection"],
#         },
#     }
# }
