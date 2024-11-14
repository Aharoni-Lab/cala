import functools as fct
import itertools as itt
import warnings
from typing import Optional, Tuple

import dask as da
import dask.array as darr
import numpy as np
import xarray as xr
from skimage.registration import phase_cross_correlation

from .utils import custom_arr_optimize, xrconcat_recursive, check_temp
from .transformation import (
    transform_perframe,
    get_bspline_param,
    get_mesh_size,
)


def estimate_motion(
    varr: xr.DataArray,
    dim: str = "frame",
    npart: int = 3,
    chunk_nfm: Optional[int] = None,
    **kwargs
) -> xr.DataArray:
    """
    Estimate motion for each frame of the input movie data.
    This function estimates motion using a recursive approach...
    """
    # Function body remains the same
    # ...


def est_motion_part(
    varr: darr.Array, npart: int, chunk_nfm: int, alt_error=5, **kwargs
) -> Tuple[darr.Array, darr.Array]:
    """
    Construct Dask graph for the recursive motion estimation algorithm.
    """
    # Function body remains the same
    # ...


def est_motion_chunk(
    varr: np.ndarray,
    sh_org: Optional[np.ndarray],
    npart: int,
    alt_error: float,
    aggregation="mean",
    upsample=100,
    max_sh=100,
    circ_thres: Optional[float] = None,
    mesh_size: Optional[Tuple[int, int]] = None,
    niter=100,
    bin_thres: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Carry out motion estimation per chunk.
    """
    # Function body remains the same
    # ...


def est_motion_perframe(
    src: np.ndarray,
    dst: np.ndarray,
    upsample: int,
    src_ma: Optional[np.ndarray] = None,
    dst_ma: Optional[np.ndarray] = None,
    mesh_size: Optional[Tuple[int, int]] = None,
    niter=100,
) -> np.ndarray:
    """
    Estimate motion given two frames.
    """
    # Function body remains the same
    # ...
