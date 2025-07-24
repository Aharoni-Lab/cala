import cv2
import numpy as np
import xarray as xr
from noob.node import Node

from cala.models import Footprints


class Footprinter(Node):

    boundary_expansion_pixels: int | None = None
    """
    Number of pixels to explore the boundary of the footprint outside of the current footprint.
    """

    tolerance: float = 1e-7

    footprints_: Footprints = None

    def ingest_frame(
        self, footprints: Footprints, pixel_stats: xr.DataArray, component_stats: xr.DataArray
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
            footprints (Footprints): Current spatial footprints Ã = [A, b].
                Shape: (pixels × components)
            pixel_stats (PixelStats): Sufficient statistics W.
                Shape: (pixels × components)
            component_stats (ComponentStats): Sufficient statistics M.
                Shape: (components × components)
        """
        A = footprints
        M = component_stats
        side_length = min(
            footprints.sizes[self.params.spatial_dims[0]],
            footprints.sizes[self.params.spatial_dims[1]],
        )
        if self.params.boundary_expansion_pixels:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_CROSS,
                (
                    self.params.boundary_expansion_pixels * 2 + 1,
                    self.params.boundary_expansion_pixels * 2 + 1,
                ),
            )  # faster than np.ones

        converged = False
        count = 0
        while not converged:
            count += 1
            mask = A > 0
            if self.params.boundary_expansion_pixels and count < side_length:
                mask = xr.apply_ufunc(
                    lambda x: cv2.morphologyEx(x, cv2.MORPH_DILATE, kernel, iterations=1),
                    mask.astype(np.uint8),
                    input_core_dims=[[*self.params.spatial_dims]],
                    output_core_dims=[[*self.params.spatial_dims]],
                    vectorize=True,
                    dask="parallelized",
                )
            # Compute AM product using xarray operations
            # Reshape M to align dimensions for broadcasting
            AM = (A @ M).rename({f"{self.params.component_dim}'": f"{self.params.component_dim}"})
            numerator = pixel_stats - AM

            # Compute update using vectorized operations
            # Expand M diagonal for broadcasting
            M_diag = xr.apply_ufunc(
                np.diag,
                component_stats,
                input_core_dims=[component_stats.dims],
                output_core_dims=[[self.params.component_dim]],
                dask="allowed",
            )

            # Apply update equation with masking
            update = numerator / M_diag
            A_new = xr.where(mask, A + update, A)
            A_new = xr.where(A_new > 0, A_new, 0)
            if abs((A - A_new).sum() / np.prod(A.shape)) < self.params.tolerance:
                A = A_new
                converged = True
            else:
                A = A_new

        self.footprints_.array = A
        return self.footprints_
