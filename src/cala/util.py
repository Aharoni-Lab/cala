from collections.abc import Sequence
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import xarray as xr
from numpydantic.ndarray import NDArray
from sparse import COO


def create_id() -> str:
    return uuid4().hex


def combine_attr_replaces(attrs: Sequence[dict[str, list[str]]], context: None = None) -> dict:
    repl = {item for attr in attrs for item in attr.get("replaces", [])}
    return {"replaces": list(repl)} if repl else {}


def clear_dir(directory: Path | str) -> None:
    for path in Path(directory).glob("**/*"):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            rmtree(path)


def sp_matmul(
    left: xr.DataArray, dim: str, rename_map: dict, right: xr.DataArray | None = None
) -> xr.DataArray:
    """
    Faster than xarray @ (for sparse arrays). The syntax is complicated enough that I'm making a
    utility function

    :param left:
    :param dim:
    :param rename_map:
    :param right:
    """

    ll = stack_sparse(left, dim).tocsr()

    if right is None:
        right = left
        rr = ll
    else:
        rr = stack_sparse(right, dim).tocsr()

    val = ll @ rr.T

    return xr.DataArray(
        COO.from_scipy_sparse(val), dims=[dim, f"{dim}'"], coords=left[dim].coords
    ).assign_coords(right[dim].rename(rename_map).coords)


def stack_sparse(da: xr.DataArray, dim: str) -> NDArray:
    return da.transpose(dim, ...).data.reshape((da.sizes[dim], -1))
