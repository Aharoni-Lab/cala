import cv2
import numpy as np
import xarray as xr
from noob import process_method

from cala.assets import CompStats, Footprint, Footprints, PixStats
from cala.models import AXIS


class Footprinter:

    def __init__(self, boundary_expansion_pixels: int | None = None, tolerance: float = 1e-7):
        self.bep = boundary_expansion_pixels
        """
        Number of pixels to explore the boundary of the footprint outside of the current footprint.
        """

        self.tol = tolerance

    @process_method
    def ingest_frame(
        self, footprints: Footprints, pixel_stats: PixStats, component_stats: CompStats
    ) -> Footprints:
        """
        Update spatial footprints using sufficient statistics.

            Ã[p, i] = max(Ã[p, i] + (W[p, i] - Ã[p, :]M[i, :])/M[i, i], 0)

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

        converged = False
        expanded = False
        kernel = None, None

        while not converged:
            mask = A > 0

            if self.bep:
                kernel = kernel if kernel else self._expansion_kernel()

                if not expanded:
                    mask = self._expand_boundary(kernel, mask)
                    expanded = True

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

            # Apply update equation with masking
            update = numerator / M_diag
            A_new = mask * (A + update)
            A_new = xr.where(A_new > 0, A_new, 0)

            if abs((A - A_new).sum() / np.prod(A.shape)) < self.tol:
                A = A_new
                converged = True
            else:
                A = A_new

        footprints.array = A
        return footprints

    def _expansion_kernel(self) -> np.ndarray:
        return cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (
                self.bep * 2 + 1,
                self.bep * 2 + 1,
            ),
        )  # faster than np.ones

    def _expand_boundary(self, kernel: np.ndarray, mask: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            lambda x: cv2.morphologyEx(x, cv2.MORPH_DILATE, kernel, iterations=1),
            mask.astype(np.uint8),
            input_core_dims=[[*AXIS.spatial_dims]],
            output_core_dims=[[*AXIS.spatial_dims]],
            vectorize=True,
            dask="parallelized",
        )


def ingest_component(footprints: Footprints, new_footprint: Footprint | Footprints) -> Footprints:
    if new_footprint.array is None:
        return footprints

    if footprints.array is None:
        footprints.array = new_footprint.array.volumize.dim_with_coords(
            dim=AXIS.component_dim, coords=[AXIS.id_coord, AXIS.confidence_coord]
        )
        return footprints

    footprints.array = xr.concat([footprints.array, new_footprint.array], dim=AXIS.component_dim)
    return footprints
