from sklearn.pipeline import make_pipeline

from cala.data_io import DataIO
from cala.preprocess import Downsampler, Denoiser, GlowRemover, BackgroundEraser
from cala.motion_correction import RigidTranslator


def main():
    io = DataIO(video_paths=["video_1", "video_2"])
    io.save_raw_video("path_to_interim", "data_name")

    data = io.load_data(stage="init")

    core_axes = ["height", "width"]
    iter_axis = "frames"
    video_dimensions = tuple(core_axes + [iter_axis])

    downsampler = Downsampler(dimensions=video_dimensions, strides=(1, 1, 2))
    denoiser = Denoiser(method="median", core_axes=core_axes, kwargs={"ksize": 7})
    glow_remover = GlowRemover(iter_axis=iter_axis)
    background_eraser = BackgroundEraser(core_axes=core_axes)

    preprocessor = make_pipeline(downsampler, denoiser, glow_remover, background_eraser)

    preprocessed_data = preprocessor.fit_transform(data)

    rigid_translator = RigidTranslator(core_axes=core_axes, iter_axis=iter_axis)
    motion_corrector = make_pipeline(rigid_translator)

    motion_corrected_data = motion_corrector.fit_transform(preprocessed_data)
