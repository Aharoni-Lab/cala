from sklearn.pipeline import make_pipeline
from cala.preprocess import Downsampler, Denoiser, GlowRemover, BackgroundEraser


def main():
    downsampler = Downsampler()
    denoiser = Denoiser(method="median", kwargs={"ksize": 7})
    glow_remover = GlowRemover()
    background_eraser = BackgroundEraser()

    preprocessor = make_pipeline(downsampler, denoiser, glow_remover, background_eraser)
