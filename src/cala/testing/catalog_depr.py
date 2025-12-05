"""
Deprecated: Only used for testing to make sure refactored codes elicit the same results.

"""

import numpy as np
import xarray as xr
from noob.node import Node
from pydantic import Field
from scipy.ndimage import gaussian_filter1d

from cala.arrays import AXIS


class CatalogerDepr(Node):
    smooth_kwargs: dict
    age_limit: int
    """Don't merge with new components if older than this number of frames."""
    merge_threshold: float
    val_threshold: float = Field(gt=0, lt=1)
    cnt_threshold: int = Field(gt=0)
    """must have cnt-number of pixels that are above the val-value"""

    def _merge_matrix(
        self,
        fps: xr.DataArray,
        trs: xr.DataArray,
        fps_base: xr.DataArray | None = None,
        trs_base: xr.DataArray | None = None,
    ) -> xr.DataArray:
        fps = fps.stack(pixels=AXIS.spatial_dims)
        trs = xr.DataArray(
            gaussian_filter1d(trs.transpose(AXIS.component_dim, ...), **self.smooth_kwargs),
            dims=trs.dims,
            coords=trs.coords,
        )

        if fps_base is None:
            fps_base = fps.rename({AXIS.component_dim: AXIS.duplicate(AXIS.component_dim)})
            trs_base = trs.rename({AXIS.component_dim: AXIS.duplicate(AXIS.component_dim)})
        else:
            fps_base = fps_base.stack(pixels=AXIS.spatial_dims).rename(AXIS.component_rename)
            trs_base = xr.DataArray(
                gaussian_filter1d(
                    trs_base.transpose(AXIS.component_dim, ...), **self.smooth_kwargs
                ),
                dims=trs_base.dims,
                coords=trs_base.coords,
            )
            trs_base = trs_base.rename(AXIS.component_rename)

        overlaps = np.matmul(fps.data, fps_base.data.T) > 0
        # corr is fast. (~1ms to 4ms)
        corrs = xr.corr(trs, trs_base, dim=AXIS.frame_dim) > self.merge_threshold
        return xr.DataArray(overlaps * corrs.values, dims=corrs.dims, coords=corrs.coords)
