from typing import Annotated as A

import numpy as np
import xarray as xr
from noob import Name, process_method
from scipy.sparse.csgraph import connected_components

from cala.assets import Footprints, Frame, Overlaps, PopSnap, Traces
from cala.logging import init_logger
from cala.models import AXIS


class FrameUpdate:
    logger = init_logger(__name__)

    def __init__(self, tol: float, max_iter: int | None = None) -> None:
        self.tol = tol
        self.max_iter = max_iter

    @process_method
    def ingest_frame(
        self, traces: Traces, footprints: Footprints, frame: Frame, overlaps: Overlaps
    ) -> A[PopSnap, Name("latest_trace")]:
        """
        Update temporal traces using current spatial footprints and frame data.

        This method implements the iterative block coordinate descent update of temporal
        traces with guaranteed convergence under non-negativity constraints. It processes
        components based on their overlap relationships, ensuring that overlapping components
        are updated together for proper convergence.

        Follows the iterative formula:

            c[G_i] = max(c[G_i] + (u[G_i] - V[G_i,:]c)/v[G_i], 0)

        where:
            - c is the temporal traces vector
            - G_i represents component groups
            - u = Ã^T y (projection of current frame)
            - V = Ã^T Ã (gram matrix of spatial components)
            - v = diag{V} (diagonal elements for normalization)

        Args:
            footprints (Footprints): Spatial footprints of all components.
                Shape: (components × height × width)
            frame (xr.DataArray): Current frame data.
                Shape: (height × width)
            overlaps (Overlaps): Sparse matrix indicating component overlaps.
                Shape: (components × components), where entry (i,j) is 1 if
                components i and j overlap, and 0 otherwise.
        """
        if footprints.array is None:
            return PopSnap()

        # Prepare inputs for the update algorithm
        A = footprints.array.stack({"pixels": AXIS.spatial_dims})
        y = frame.array.stack({"pixels": AXIS.spatial_dims})
        c = traces.array.isel({AXIS.frames_dim: -1})

        _, labels = connected_components(
            csgraph=overlaps.array.data, directed=False, return_labels=True
        )
        clusters = [np.where(labels == label)[0] for label in np.unique(labels)]
        updated_traces = self._update_traces(A, y, c.copy(), clusters)

        if traces.zarr_path:
            updated_tr = updated_traces.expand_dims(AXIS.frames_dim).assign_coords(
                {
                    AXIS.timestamp_coord: (
                        AXIS.frames_dim,
                        [updated_traces[AXIS.timestamp_coord].values],
                    )
                }
            )
            traces.update(updated_tr, append_dim=AXIS.frames_dim)
        else:
            traces.array = xr.concat([traces.array, updated_traces], dim=AXIS.frames_dim)

        return PopSnap.from_array(updated_traces)

    def _update_traces(
        self,
        A: xr.DataArray,
        y: xr.DataArray,
        c: xr.DataArray,
        clusters: list[np.ndarray],
    ) -> xr.DataArray:
        """
        Implementation of the temporal traces update algorithm.

        This function uses block coordinate descent to update temporal traces
        for overlapping components together while maintaining non-negativity constraints.

        Args:
            A (xr.DataArray): Spatial footprints matrix [A, b].
                Shape: (components × pixels)
            y (xr.DataArray): Current data frame.
                Shape: (pixels,)
            c (xr.DataArray): Last value of temporal traces. (just used for shape)
                Shape: (components,)
            clusters (list[np.ndarray]): list of groups that each contain component indices that
                have overlapping footprints.

        Returns:
            xr.DataArray: Updated temporal traces satisfying non-negativity constraints.
                Shape: (components,)
        """
        # Step 1: Compute projection of current frame
        u = (A @ y).as_numpy()

        # Step 2: Compute gram matrix of spatial components
        V = (A @ A.rename({AXIS.component_dim: f"{AXIS.component_dim}'"})).as_numpy()

        # Step 3: Extract diagonal elements for normalization
        V_diag = np.diag(V)

        cnt = 0

        # Steps 4-9: Main iteration loop until convergence
        while True:
            c_old = c.copy()

            # Steps 6-8: Update each group using block coordinate descent
            for cluster in clusters:
                # Update traces for current group (division is pointwise)

                numerator = u.isel({AXIS.component_dim: cluster}) - (
                    V.isel({f"{AXIS.component_dim}'": cluster}) @ c
                ).rename({f"{AXIS.component_dim}'": AXIS.component_dim})

                c.loc[{AXIS.component_dim: cluster}] = np.maximum(
                    c.isel({AXIS.component_dim: cluster}) + numerator / V_diag[cluster].T, 0
                )

            cnt += 1
            maxed = self.max_iter and (cnt == self.max_iter)

            if np.linalg.norm(c - c_old) >= self.tol * np.linalg.norm(c_old) or maxed:
                if maxed:
                    self.logger.debug(msg="max_iter reached before converging.")
                return xr.DataArray(
                    c.values, dims=c.dims, coords=c[AXIS.component_dim].coords
                ).assign_coords(y[AXIS.frames_dim].coords)


def ingest_component(traces: Traces, new_traces: Traces) -> Traces:
    """

    :param traces:
    :param new_traces: Can be either a newly registered trace or an updated existing one.
    :return:
    """

    c = traces.full_array()
    c_det = new_traces.array

    if c_det is None:
        return traces

    if c is None:
        traces.array = c_det
        return traces

    if c.sizes[AXIS.frames_dim] > c_det.sizes[AXIS.frames_dim]:
        # if newly detected cells are truncated
        c_new = xr.DataArray(
            np.full((c_det.sizes[AXIS.component_dim], c.sizes[AXIS.frames_dim]), np.nan),
            dims=[AXIS.component_dim, AXIS.frames_dim],
            coords=c.isel({AXIS.component_dim: 0}).coords,
        )
        c_new[AXIS.id_coord] = c_det[AXIS.id_coord]
        c_new[AXIS.detect_coord] = c_det[AXIS.detect_coord]

        c_new.loc[{AXIS.frames_dim: c_det[AXIS.frame_coord]}] = c_det
    else:
        c_new = c_det.sel({AXIS.frame_coord: c[AXIS.frame_coord]})

    merged_ids = c_det.attrs.get("replaces")
    if merged_ids:
        if traces.zarr_path:
            invalid = c[AXIS.id_coord].isin(merged_ids)
            traces.array = c.where(~invalid.compute(), drop=True).compute()
        else:
            intact_ids = [id_ for id_ in c[AXIS.id_coord].values if id_ not in merged_ids]
            c = (
                c.set_xindex(AXIS.id_coord)
                .sel({AXIS.id_coord: intact_ids})
                .reset_index(AXIS.id_coord)
            )

    if traces.zarr_path:
        traces.update(c_new, append_dim=AXIS.component_dim)
    else:
        traces.array = xr.concat([c, c_new], dim=AXIS.component_dim, combine_attrs="drop")

    return traces
