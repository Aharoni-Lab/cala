from typing import Annotated as A
import cv2
import numpy as np
import xarray as xr
from noob import process_method, Name

from cala.assets import CompStats, Footprint, Footprints, PixStats
from cala.models import AXIS


class Footprinter:

    def __init__(self, tol: float, max_iter: int | None = None, bep: int | None = None):
        self.bep = bep
        """
        Number of pixels to explore the boundary of the footprint outside of the current footprint.
        """

        self.tol = tol

        self.max_iter = max_iter

    @process_method
    def ingest_frame(
        self, footprints: Footprints, pixel_stats: PixStats, component_stats: CompStats
    ) -> A[Footprints, Name("footprints")]:
        """
        Update spatial footprints using sufficient statistics.

            Ã[p, i] = max(Ã[p, i] + (W[p, i] - Ã[p, :]M[:, i])/M[i, i], 0)

        where:
            - Ã is the spatial footprints matrix
            - W is the pixel-wise sufficient statistics
            - M is the component-wise sufficient statistics
            - p are the pixels where component i can be non-zero

        Args:
            pixel_stats (PixelStats): Sufficient statistics W.
                Shape: (pixels × components)
            component_stats (ComponentStats): Sufficient statistics M.
                Shape: (components × components)
        """
        if footprints.array is None:
            return footprints

        A = footprints.array
        M = component_stats.array
        W = pixel_stats.array

        mask = A > 0

        if self.bep:
            kernel = self._expansion_kernel()
            mask = self._expand_boundary(kernel, mask)

        # for _ in range(self.max_iter):
        while True:
            AM = A.rename(AXIS.component_rename) @ M
            numerator = W - AM

            # Expand M diagonal for broadcasting
            M_diag = xr.apply_ufunc(
                np.diag,
                M,
                input_core_dims=[M.dims],
                output_core_dims=[[AXIS.component_dim]],
                dask="allowed",
            )

            update = numerator / (M_diag + np.finfo(float).tiny)
            A_new = (mask * (A + update)).clip(min=0)

            if (np.abs(A - A_new).sum() / np.prod(A.shape)).item() < self.tol:
                footprints.array = A_new
                return footprints
            else:
                A = A_new
                mask = A > 0

    def _expansion_kernel(self) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (self.bep * 2 + 1, self.bep * 2 + 1))

    def _expand_boundary(self, kernel: np.ndarray, mask: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            lambda x: cv2.morphologyEx(x, cv2.MORPH_DILATE, kernel, iterations=1),
            mask.astype(np.uint8),
            input_core_dims=[AXIS.spatial_dims],
            output_core_dims=[AXIS.spatial_dims],
            vectorize=True,
            dask="parallelized",
        )


def ingest_component(footprints: Footprints, new_footprints: Footprints) -> Footprints:
    if new_footprints.array is None:
        return footprints

    a = footprints.array
    a_det = new_footprints.array

    if footprints.array is None:
        footprints.array = a_det
        return footprints

    merged = a_det.attrs.get("replaces")
    if merged:
        a = (
            a.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: [id_ for id_ in a[AXIS.id_coord].values if id_ not in merged]})
            .reset_index(AXIS.id_coord)
        )

    footprints.array = xr.concat([a, a_det], dim=AXIS.component_dim, combine_attrs="drop")

    return footprints
