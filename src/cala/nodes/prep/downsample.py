from typing import Annotated as A

import numpy as np
from noob import Name

from cala.arrays import AXIS, Frame
from cala.nodes.prep import package_frame


def downsample(
    frames: list[Frame], x_range: tuple[int, int], y_range: tuple[int, int], t_downsample: int = 1
) -> A[Frame, Name("frame")]:
    """
    Downsampling in time and cropping in space. Must be followed by gather node, and
    t_downsample has to be same as gather's parameter n value.

    :param frames:
    :param x_range:
    :param y_range:
    :param t_downsample:
    :return:
    """
    arrays = []
    for frame in frames:
        arrays.append(frame.array[x_range[0] : x_range[1], y_range[0] : y_range[1]])

    return package_frame(
        np.mean(arrays, axis=0), arrays[-1][AXIS.frame_coord].item() // t_downsample
    )
