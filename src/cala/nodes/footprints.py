from typing import Annotated as A

import cv2
import numpy as np
import xarray as xr
from noob import Name, process_method
from sparse import COO

from cala.assets import CompStats, Footprints, PixStats
from cala.logging import init_logger
from cala.models import AXIS


class Footprinter:
    logger = init_logger(__name__)

    def __init__(
        self,
        tol: float,
        max_iter: int | None = None,
        bep: int | None = None,
        ratio_lb: float = 0.15,
    ):
        self.bep = bep
        """
        Number of pixels to explore the boundary of the footprint outside of the current footprint.
        """

        self.ratio_lb = ratio_lb
        """
        Ratio of the least bright pixel against the brightest pixel of a given footprint.
        """

        self.tol = tol
        self.max_iter = max_iter

    @process_method
    def ingest_frame(
        self, footprints: Footprints, pixel_stats: PixStats, component_stats: CompStats, index: int
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

        plain_mask = A > 0
        mask = self._build_mask(plain_mask, index=index)

        # Expand M diagonal for broadcasting
        M_diag = xr.apply_ufunc(
            np.diag,
            M,
            input_core_dims=[M.dims],
            output_core_dims=[[AXIS.component_dim]],
            dask="allowed",
        )

        cnt = 0
        while True:
            AM = A.rename(AXIS.component_rename) @ M
            numerator = W - AM

            update = numerator / (M_diag + np.finfo(float).tiny)
            A_new = (mask * (A + update)).clip(min=0)

            step = (np.abs(A - A_new).sum() / np.prod(A.shape)).item()

            cnt += 1
            maxed = self.max_iter and (cnt == self.max_iter)

            if step < self.tol or maxed:
                A_final = A_new.where(
                    A_new > A_new.max(AXIS.spatial_dims) * self.ratio_lb, 0, drop=False
                )
                if maxed:
                    self.logger.debug(msg="max_iter reached before converging.")
                    A_final = A_new.where(plain_mask, 0, drop=False)

                footprints.array = A_final
                return footprints
            else:
                A = A_new
                mask = A > 0

    def _expansion_kernel(self) -> np.ndarray:
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

    def _expand_boundary(self, kernel: np.ndarray, mask: xr.DataArray) -> xr.DataArray:
        expanded = xr.apply_ufunc(
            lambda x: cv2.morphologyEx(x, cv2.MORPH_DILATE, kernel, iterations=1),
            mask.as_numpy().astype(np.uint8),
            input_core_dims=[AXIS.spatial_dims],
            output_core_dims=[AXIS.spatial_dims],
            vectorize=True,
            dask="parallelized",
        )
        expanded.data = COO.from_numpy(expanded.data)
        return expanded

    def _build_mask(self, mask: xr.DataArray, index: int) -> xr.DataArray:
        expansion_left = (index - mask[AXIS.detect_coord] - self.bep) <= 0
        expand_ids = expansion_left.where(expansion_left, drop=True)[AXIS.id_coord].values
        no_expand_ids = expansion_left.where(~expansion_left, drop=True)[AXIS.id_coord].values

        if expand_ids.size > 0:
            kernel = self._expansion_kernel()

            expanded_mask = self._expand_boundary(
                kernel, mask.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: expand_ids})
            )

            final_mask = xr.concat(
                [
                    mask.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: no_expand_ids}),
                    expanded_mask,
                ],
                dim=AXIS.component_dim,
            ).reset_index(AXIS.id_coord)
        else:
            final_mask = mask
        return final_mask


def ingest_component(
    footprints: Footprints, new_footprints: Footprints
) -> A[Footprints, Name("footprints")]:
    if new_footprints.array is None:
        return footprints

    a = footprints.array
    a_det = new_footprints.array

    if footprints.array is None:
        footprints.array = a_det
        return footprints

    merged_ids = a_det.attrs.get("replaces")
    if merged_ids:
        intact_ids = [id_ for id_ in a[AXIS.id_coord].values if id_ not in merged_ids]
        a = a.set_xindex(AXIS.id_coord).sel({AXIS.id_coord: intact_ids}).reset_index(AXIS.id_coord)

    footprints.array = xr.concat([a, a_det], dim=AXIS.component_dim, combine_attrs="drop")

    return footprints
