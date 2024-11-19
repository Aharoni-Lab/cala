# Cala: Calcium Imaging Analysis Pipeline for Long-term Recordings

## Features
A calcium imaging pipeline focused on long-term massive recordings that is based on a Sklearn pipeline architecture. Streamlined integration into an endless list of 3rd party apps that support Scikit-learn, including but not limited to hyperparameter optimization tools (i.e. Optuna), ML pipeline management tools (i.e. MLFlow), etc. Future implementation will include interactive UI and a modular orchestration architecture that supports piecewise progress, optimized orchestration, and automatic data, artifact, and pipeline versioning.

## Requirements
Tests currently cover Python versions 3.11 and 3.12.

## Installation
https://pypi.org/project/cala/0.1.0/
```shell
pip install cala==0.1.0
```

## Usage
```python
from sklearn.pipeline import make_pipeline

from cala.data_io import DataIO
from cala.preprocess import Downsampler, Denoiser, GlowRemover, BackgroundEraser
from cala.motion_correction import RigidTranslator


def main():
    io = DataIO(video_paths=["video_1", "video_2"])
    io.save_raw_video("path_to_interim", "data_name")

    core_axes = ["height", "width"]
    iter_axis = "frames"
    video_dimensions = tuple(core_axes + [iter_axis])

    downsampler = Downsampler(dimensions=video_dimensions, strides=(1, 1, 2))
    denoiser = Denoiser(method="median", core_axes=core_axes, kwargs={"ksize": 7})
    glow_remover = GlowRemover(iter_axis=iter_axis)
    background_eraser = BackgroundEraser(core_axes=core_axes)
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
## Roadmap
EOM 11/2024: Batch processing complete
EOM 12/2024: Engineering first iteration complete
EOM 01/2025: Orchestration first integration complete
EOM 02/2025: UI first iteration complete

## Contributing
We welcome contributions! Please fork this repository and submit a pull request if you would like to contribute to the project. You can also open issues for bug reports, feature requests, or discussions.

## License

## Contact
For questions or support, please reach out to Raymond Chang at [raymond@physics.ucla.edu](mailto:raymond@physics.ucla.edu).
