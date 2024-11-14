from typing import Optional, Tuple

import numpy as np
import SimpleITK as sitk
import xarray as xr


def apply_shifts(varr: xr.DataArray, shifts: xr.DataArray, fill=np.nan) -> xr.DataArray:
    """
    Apply rigid shifts to the input movie data.

    This function applies the calculated rigid shifts to each frame...
    (Keep the original docstring)
    """
    sh_dim = shifts.coords["shift_dim"].values.tolist()
    varr_sh = xr.apply_ufunc(
        shift_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        shifts,
        input_core_dims=[sh_dim, ["shift_dim"]],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask="parallelized",
        kwargs={"fill": fill},
        output_dtypes=[varr.dtype],
    )
    return varr_sh


def shift_perframe(fm: np.ndarray, sh: np.ndarray, fill=np.nan) -> np.ndarray:
    """
    Apply a rigid shift to a single frame.
    (Keep the original docstring)
    """
    if np.isnan(fm).all():
        return fm
    sh = np.around(sh).astype(int)
    fm = np.roll(fm, sh, axis=np.arange(fm.ndim))
    for ish, s in enumerate(sh):
        index = [slice(None)] * fm.ndim
        if s > 0:
            index[ish] = slice(None, s)
            fm[tuple(index)] = fill
        elif s < 0:
            index[ish] = slice(s, None)
            fm[tuple(index)] = fill
    return fm


def apply_transform(
    varr: xr.DataArray,
    trans: xr.DataArray,
    fill=0,
    mesh_size: Optional[Tuple[int, int]] = None,
) -> xr.DataArray:
    """
    Apply necessary transform to correct for motion.

    This function can correct for both rigid and non-rigid motion...
    (Keep the original docstring)
    """
    sh_dim = trans.coords["shift_dim"].values.tolist()
    if "grid0" in trans.dims:
        fm0 = varr.isel(frame=0).values
        if mesh_size is None:
            mesh_size = get_mesh_size(fm0)
        param = get_bspline_param(fm0, mesh_size)
        mdim = ["shift_dim", "grid0", "grid1"]
    else:
        param = None
        mdim = ["shift_dim"]
    varr_sh = xr.apply_ufunc(
        transform_perframe,
        varr.chunk({d: -1 for d in sh_dim}),
        trans,
        input_core_dims=[sh_dim, mdim],
        output_core_dims=[sh_dim],
        vectorize=True,
        dask="parallelized",
        kwargs={"fill": fill, "param": param},
        output_dtypes=[varr.dtype],
    )
    return varr_sh


def transform_perframe(
    fm: np.ndarray,
    tx_coef: np.ndarray,
    fill=0,
    param: Optional[np.ndarray] = None,
    mesh_size: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """
    Transform a single frame.
    (Keep the original docstring)
    """
    if tx_coef.ndim > 1:
        if param is None:
            if mesh_size is None:
                mesh_size = get_mesh_size(fm)
            param = get_bspline_param(fm, mesh_size)
        tx = sitk.BSplineTransform(2)
        tx.SetFixedParameters(param)
        tx.SetParameters(tx_coef.flatten())
    else:
        tx = sitk.TranslationTransform(2, -tx_coef[::-1])
    fm = sitk.GetImageFromArray(fm)
    fm = sitk.Resample(fm, fm, tx, sitk.sitkLinear, fill)
    return sitk.GetArrayFromImage(fm)


def get_bspline_param(img: np.ndarray, mesh_size: Tuple[int, int]) -> np.ndarray:
    """
    Compute fixed parameters for the BSpline transform given a frame and mesh size.
    (Keep the original docstring)
    """
    return sitk.BSplineTransformInitializer(
        image1=sitk.GetImageFromArray(img),
        transformDomainMeshSize=mesh_size,
    ).GetFixedParameters()


def get_mesh_size(fm: np.ndarray) -> Tuple[int, int]:
    """
    Compute suitable mesh size given a frame.
    (Keep the original docstring)
    """
    return (
        int(np.around(fm.shape[0] / 100)),
        int(np.around(fm.shape[1] / 100)),
    )
