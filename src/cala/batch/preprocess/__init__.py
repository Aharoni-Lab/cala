"""
Preprocessing
    * downsampling
    * calculate chunk
    * glow removal
    * denoise
    * background removal
            This step attempts to estimate background (everything except the fluorescent signal of in-focus cells) and subtracts it from the frame.
            By default we use a morphological tophat operation to estimate the background from each frame:
            First, a [disk element](http://scikit-image.org/docs/dev/api/skimage.morphology.html#disk) with a radius of `wnd` is created.
            Then, a [morphological erosion](https://homepages.inf.ed.ac.uk/rbf/HIPR2/erode.htm) using the disk element is applied to each frame, which eats away any bright "features" that are smaller than the disk element.
            Subsequently, a [morphological dilation](https://homepages.inf.ed.ac.uk/rbf/HIPR2/dilate.htm) is applied to the "eroded" image, which in theory undoes the erosion except the bright "features" that were completely eaten away.
            The overall effect of this process is to remove any bright feature that is smaller than a disk with radius `wnd`.
            Thus, when setting `wnd` to the **largest** expected radius of cell, this process can give us a good estimation of the background.
            Then finally the estimated background is subtracted from each frame.
"""

from .background_erasure import BackgroundEraser
from .denoise import Denoiser
from .downsample import Downsampler
from .glow_removal import GlowRemover

__all__ = [
    "Downsampler",
    "Denoiser",
    "GlowRemover",
    "BackgroundEraser",
]
