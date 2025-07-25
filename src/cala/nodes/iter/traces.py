import numpy as np
import xarray as xr
from noob.node import Node
from numba import prange
from scipy.sparse.csgraph import connected_components

from cala.models import AXIS, Footprints, Frame, Movie, Overlap, Traces


class Tracer(Node):
    tolerance: float = 1e-3

    traces_: Traces = None

    def process(
        self,
        footprints: Footprints,
        movie: Movie = None,
        frame: Frame = None,
        traces: Traces = None,
        overlaps: Overlap = None,
    ) -> Traces:
        """
        A jenky ass temporary process method to circumvent Resource not yet being implemented.
        """
        if movie is None:
            return self.ingest_frame(
                footprints=footprints, frame=frame, traces=traces, overlaps=overlaps
            )
        else:
            return self.initialize(footprints=footprints, movie=movie)

    def initialize(self, footprints: Footprints, movie: Movie) -> Traces:
        """Learn temporal traces from footprints and frames."""
        A = footprints.array
        Y = movie.array

        # Get frames to use and flatten them
        n_frames = Y.sizes[AXIS.frames_dim]
        flattened_frames = Y[:n_frames].stack({"pixels": AXIS.spatial_dims})
        flattened_footprints = A.stack({"pixels": AXIS.spatial_dims})

        # Process all components
        temporal_traces = self.solve_all_component_traces(
            flattened_footprints.values,
            flattened_frames.values,
            flattened_footprints.sizes[AXIS.component_dim],
            flattened_frames.sizes[AXIS.frames_dim],
        )

        trace_coords = [
            AXIS.id_coord,
            AXIS.confidence_coord,
            AXIS.frame_coord,
            AXIS.timestamp_coord,
        ]
        coords = {k: v for k, v in {**A.coords, **Y.coords}.items() if k in trace_coords}
        self.traces_ = Traces(
            array=xr.DataArray(
                temporal_traces,
                dims=(AXIS.component_dim, AXIS.frames_dim),
                coords=coords,
            )
        )

        return self.traces_

    def ingest_frame(
        self, footprints: Footprints, frame: Frame, traces: Traces, overlaps: Overlap
    ) -> Traces:
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
            traces (Traces): Current temporal traces to be updated.
                Shape: (components × time)
            overlaps (Overlaps): Sparse matrix indicating component overlaps.
                Shape: (components × components), where entry (i,j) is 1 if
                components i and j overlap, and 0 otherwise.
        """
        # Prepare inputs for the update algorithm
        A = footprints.array.stack({"pixels": AXIS.spatial_dims})
        y = frame.array.stack({"pixels": AXIS.spatial_dims})
        c = traces.array.isel({AXIS.frames_dim: [-1]})

        _, labels = connected_components(csgraph=overlaps.data, directed=False, return_labels=True)
        clusters = [np.where(labels == label)[0] for label in np.unique(labels)]

        updated_traces = self.update_traces(A, y, c.copy(), clusters, self.tolerance)

        self.traces_.array = updated_traces

        return self.traces_

    def update_traces(
        self,
        A: xr.DataArray,
        y: xr.DataArray,
        c: xr.DataArray,
        clusters: list[np.ndarray],
        eps: float,
    ) -> xr.DataArray:
        """
        Implementation of the temporal traces update algorithm.

        This function implements the core update logic. It uses block coordinate descent
        to update temporal traces for overlapping components together while maintaining
        non-negativity constraints.

        Args:
            A (xr.DataArray): Spatial footprints matrix [A, b].
                Shape: (components × pixels)
            y (xr.DataArray): Current data frame.
                Shape: (pixels,)
            c (xr.DataArray): Last value of temporal traces. (just used for shape)
                Shape: (components,)
            clusters (list[np.ndarray]): list of groups that each contain component indices that
                have overlapping footprints.
            eps (float): Tolerance level for convergence checking.

        Returns:
            xr.DataArray: Updated temporal traces satisfying non-negativity constraints.
                Shape: (components,)
        """
        # Step 1: Compute projection of current frame
        u = A @ y

        # Step 2: Compute gram matrix of spatial components
        V = A @ A.rename({AXIS.component_dim: f"{AXIS.component_dim}'"})

        # Step 3: Extract diagonal elements for normalization
        V_diag = np.diag(V)

        # Step 4: Initialize previous iteration value
        c_old = np.zeros_like(c)

        # Steps 5-10: Main iteration loop until convergence
        while np.linalg.norm(c - c_old) >= eps * np.linalg.norm(c_old):
            c_old = c.copy()

            # Steps 7-9: Update each group using block coordinate descent
            for cluster in clusters:
                # Update traces for current group (division is pointwise)
                numerator = u.isel({AXIS.component_dim: cluster}) - (
                    V.isel({f"{AXIS.component_dim}'": cluster}) @ c
                ).rename({f"{AXIS.component_dim}'": AXIS.component_dim})

                c.loc[{AXIS.component_dim: cluster}] = np.maximum(
                    c.isel({AXIS.component_dim: cluster})
                    + numerator / np.array([V_diag[cluster]]).T,
                    0,
                )

        return c

    @staticmethod
    def solve_all_component_traces(
        footprints: np.ndarray, frames: np.ndarray, n_components: int, n_frames: int
    ) -> np.ndarray:
        """Solve temporal traces for all components in parallel

        Args:
            footprints: Array of shape (n_components, height*width)
            frames: Array of shape (n_frames, height*width)
        Returns:
            Array of shape (n_components, n_frames)
        """
        results = np.zeros((n_components, n_frames), dtype=frames.dtype)

        # Parallel loop over components
        for i in prange(n_components):
            footprint = footprints[i]
            active_pixels = footprint > 0

            if np.any(active_pixels):
                footprint_active = footprint[active_pixels]
                frames_active = frames[:, active_pixels]
                results[i] = Tracer.fast_nnls_vector(footprint_active, frames_active)

        return results

    @staticmethod
    def fast_nnls_vector(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Specialized NNLS for single-variable case across multiple frames
        A: footprint values (n_pixels,)
        B: frame data matrix (n_frames, n_pixels)
        Returns: brightness values for each frame (n_frames,)
        """
        ata = (A * A).sum()  # Compute once for all frames
        if ata <= 0:
            return np.zeros(B.shape[0], dtype=B.dtype)

        # Vectorized computation for all frames
        atb = A @ B.T  # dot product with each frame
        return np.maximum(0, atb / ata)
