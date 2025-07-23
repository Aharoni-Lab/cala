from typing import Literal, Callable, Any

import cv2
import numpy as np
import pytest
import xarray as xr

from cala.models import AXIS
from cala.nodes.prep.denoise import denoise
from cala.testing.toy import Toy, Position, FrameDims


@pytest.mark.parametrize(
    "method, func, params",
    [
        ("gaussian", cv2.GaussianBlur, {"ksize": (3, 3), "sigmaX": 0.3}),
        ("median", cv2.medianBlur, {"ksize": 3}),
        ("bilateral", cv2.bilateralFilter, {"d": 3, "sigmaColor": 75, "sigmaSpace": 75}),
    ],
)
def test_denoise(
    method: Literal["gaussian", "median", "bilateral"], func: Callable, params: dict[str, Any]
):
    toy = Toy(
        n_frames=10,
        frame_dims=FrameDims(width=11, height=11),
        cell_radii=1,
        cell_positions=[Position(width=5, height=5)],
        cell_traces=[np.ones(10)],
    )

    gen = toy.movie_gen()
    movie = toy.make_movie()

    expected = xr.apply_ufunc(
        func,
        movie.array.astype(np.float32),  # cv2.medianBlur does not work with float64
        input_core_dims=[AXIS.spatial_dims],
        output_core_dims=[AXIS.spatial_dims],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[movie.array.dtype],
        kwargs=params,
    )

    results = []
    for frame in iter(gen):
        results.append(denoise(frame=frame, method=method, **params))

    for exp, res in zip(expected, results):
        np.testing.assert_allclose(exp.values, res.array.values)
