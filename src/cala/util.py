from collections.abc import Sequence
from pathlib import Path
from shutil import rmtree
from uuid import uuid4

import numpy as np
import xarray as xr


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

    left = left.transpose(dim, ...)

    if right is None:
        right = left
    else:
        right = right.transpose(dim, ...)
    val = np.matmul(
        np.reshape(left.data, (left.sizes[dim], -1)),
        np.reshape(right.data, (right.sizes[dim], -1)).T,
    )
    return xr.DataArray(val, dims=[dim, f"{dim}'"], coords=left[dim].coords).assign_coords(
        right[dim].rename(rename_map).coords
    )
