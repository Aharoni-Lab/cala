import cv2
import numpy as np
import xarray as xr
from noob.node import Node

from cala.models import PixStat, Footprints, CompStat, AXIS, Footprint


class Footprinter(Node):

    boundary_expansion_pixels: int | None = None
    """
    Number of pixels to explore the boundary of the footprint outside of the current footprint.
    """

    tolerance: float = 1e-7

    footprints_: Footprints = None

    def process(self, pixel_stats: PixStat, component_stats: CompStat) -> Footprints:
        return self.ingest(pixel_stats, component_stats)

    def ingest_frame(self, pixel_stats: PixStat, component_stats: CompStat) -> Footprints:
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
        A = self.footprints_.array
        M = component_stats.array
        W = pixel_stats.array

        converged = False
        expanded = False
        kernel = None, None

        while not converged:
            mask = A > 0

            if self.boundary_expansion_pixels:
                kernel = self.expansion_kernel() if not kernel else kernel

                if not expanded:
                    mask = self.expand_boundary(kernel, mask)
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

            if abs((A - A_new).sum() / np.prod(A.shape)) < self.tolerance:
                A = A_new
                converged = True
            else:
                A = A_new

        self.footprints_.array = A
        return self.footprints_

    def expansion_kernel(self) -> np.ndarray:
        return cv2.getStructuringElement(
            cv2.MORPH_CROSS,
            (
                self.boundary_expansion_pixels * 2 + 1,
                self.boundary_expansion_pixels * 2 + 1,
            ),
        )  # faster than np.ones

    def expand_boundary(self, kernel: np.ndarray, mask: xr.DataArray) -> xr.DataArray:
        return xr.apply_ufunc(
            lambda x: cv2.morphologyEx(x, cv2.MORPH_DILATE, kernel, iterations=1),
            mask.astype(np.uint8),
            input_core_dims=[[*AXIS.spatial_dims]],
            output_core_dims=[[*AXIS.spatial_dims]],
            vectorize=True,
            dask="parallelized",
        )

    def ingest_component(self, new_footprint: Footprint | Footprints) -> Footprints:
        self.footprints_.array = xr.concat(
            [self.footprints_.array, new_footprint.array], dim=AXIS.component_dim
        )
        return self.footprints_
