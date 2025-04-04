# Cala: Calcium Imaging Analysis Pipeline for Long-term Recordings

[![PyPI - Version](https://img.shields.io/pypi/v/cala)](https://pypi.org/project/cala/)
![PyPI - Python Version](https://img.shields.io/pypi/pyversions/cala)
![PyPI - Status](https://img.shields.io/pypi/status/cala)
[![codecov](https://codecov.io/gh/Aharoni-Lab/cala/graph/badge.svg?token=Apn4YtSvbU)](https://codecov.io/gh/Aharoni-Lab/cala)

## Features

A calcium imaging pipeline focused on long-term massive recordings that is based on a [**Sklearn
**](https://scikit-learn.org/stable/) / [**River**](https://riverml.xyz/latest/) pipeline architecture.

Streamlined integration into an endless list of 3rd party apps that support Scikit-learn, including but not limited to
hyperparameter optimization tools (i.e. Optuna), ML pipeline management tools (i.e. MLFlow), etc.

Streaming side incorporates real-time visualizations and parameter updates.

Future implementation will include interactive UI and a modular orchestration architecture that supports piecewise
progress, optimized orchestration, and automatic data, artifact, and pipeline versioning.

## Requirements

Tests currently cover Python versions 3.11 and 3.12.

## Installation

https://pypi.org/project/cala/0.1.0/

```shell
pip install cala==0.1.0
```

## Usage

### Batch

```python
from sklearn.pipeline import make_pipeline

from cala.batch.preprocess import Downsampler, Denoiser, GlowRemover, BackgroundEraser
from cala.batch.video_stabilization import RigidTranslator


def main():
    io = DataIO(video_paths=["video_1", "video_2"])
    io.save_raw_video("path_to_interim", "data_name")

    core_axes = ["height", "width"]
    iter_axis = "frames"
    video_dimensions = tuple(core_axes + [iter_axis])

    downsampler = Downsampler(dimensions=video_dimensions, strides=(1, 1, 2))
    denoiser = Denoiser(method="median", core_axes=core_axes, kwargs={"ksize": 7})
    glow_remover = GlowRemover(iter_axis=iter_axis)

    preprocessor = make_pipeline(downsampler, denoiser, glow_remover, background_eraser)

    rigid_translator = RigidTranslator(core_axes=core_axes, iter_axis=iter_axis)

    motion_corrector = make_pipeline(rigid_translator)

    data = io.load_data(stage="init")

    # Option 1:
    preprocessed_data = preprocessor.fit_transform(data)
    motion_corrected_data = motion_corrector.fit_transform(preprocessed_data)
    demixed_data = demixer.fit_transform(motion_corrected_data)
    deconvolved_data = deconvolver.fit_transform(demixed_data)

    # Option 2:
    cala_pipeline = make_pipeline(preprocessor, motion_corrector, demixer, deconvolver)
    deconvolved_data = cala_pipeline.fit_transform(data)


if __name__ == "__main__":
    main()
```

### Stream

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
    ...
},
```

the actual python main.py looks like the following (will be eventually implemented as a cli `run` command:

```python
runner = Runner(streaming_config)
video, _, _ = raw_calcium_video

for idx, frame in enumerate(video):
    frame = Frame(frame, idx)
    frame = runner.preprocess(frame)

    if not runner.is_initialized:
        runner.initialize(frame)
        continue

    runner.iterate(frame)
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
