# Cala

[![PyPI - Version](https://img.shields.io/pypi/v/cala)](https://pypi.org/project/cala/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cala)
![PyPI - Status](https://img.shields.io/pypi/status/cala)
[![codecov](https://codecov.io/gh/Aharoni-Lab/cala/graph/badge.svg?token=Apn4YtSvbU)](https://codecov.io/gh/Aharoni-Lab/cala)

## Features

A calcium imaging pipeline focused on long-term massive recordings.

## Requirements

Tests currently cover Python versions 3.11, 3.12 and 3.13.

## Installation

https://pypi.org/project/cala/0.1.0/

```shell
pip install cala==0.1.0
```

## Usage

Due to various reasons, the streaming side is structured in a graph-&-state based manner rather than a linear pipeline.
This accompanies a config file that maps how the transformations are related to each other, and a short python code that
actually runs the configured plan. The design schematic can be viewed here: (*
*[link](https://lucid.app/documents/embedded/808097f9-bf66-4ea8-9df0-e957e6bd0931)**)

#### Config

```python
"preprocess": {
    "downsample": {
        "transformer": Downsampler,
        "params": {
            "method": "mean",
            "dimensions": ["width", "height"],
            "strides": [2, 2],
        },
    },
    ...
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
    ...
},
"iteration": {
    "traces": {
        "transformer": TracesUpdater,
        "params": {"tolerance": 1e-3},
    },
    ...
},
```

the actual python main.py looks like the following (will be eventually implemented as a cli `run` command):

```python
runner = Runner(streaming_config)

for idx, frame in enumerate(stream):
    frame = runner.preprocess(frame)

    if not runner.is_initialized:
        runner.initialize(frame)
        continue

    runner.iterate(frame)

dump(runner._state)
```

## Roadmap

*EOM 03/2025:* Streaming first iteration complete\
*EOM 04/2025:* UI first iteration complete

## Contributing

We welcome contributions! Please fork this repository and submit a pull request if you would like to contribute to the
project. You can also open issues for bug reports, feature requests, or discussions.

## Test Coverage Status

https://app.codecov.io/gh/Aharoni-Lab/cala

## License

## Contact

For questions or support, please reach out to Raymond Chang
at [raymond@physics.ucla.edu](mailto:raymond@physics.ucla.edu).
