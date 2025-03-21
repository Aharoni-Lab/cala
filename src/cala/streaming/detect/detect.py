from dataclasses import dataclass
from typing import Self, Tuple, List

import numpy as np
import sparse
import xarray as xr
from river.base import SupervisedTransformer
from scipy.ndimage import gaussian_filter
from sklearn.exceptions import NotFittedError

from cala.streaming.core import Parameters, Footprints, Traces
from cala.streaming.stores.odl import Residuals, PixelStats, ComponentStats, Overlaps


@dataclass
class DetectNewComponentsParams(Parameters):
    """Parameters for new component detection.

    This class defines the configuration parameters needed for detecting new
    components from residual signals, including thresholds for spatial and
    temporal correlations, and filtering parameters for spatial processing.
    """

    component_axis: str = "components"
    """Name of the dimension representing individual components."""

    frames_axis: str = "frames"
    """Name of the dimension representing time points."""

    spatial_axes: tuple = ("height", "width")
    """Names of the dimensions representing spatial coordinates (height, width)."""

    gaussian_radius: float = 2.0
    """Radius (τ) of Gaussian kernel for spatial filtering."""

    spatial_threshold: float = 0.8
    """Threshold for correlation in space (r_s)."""

    temporal_threshold: float = 0.8
    """Threshold for correlation in time (r_t)."""

    def validate(self):
        """Validate parameter configurations.

        Raises:
            ValueError: If gaussian_radius is not positive or if thresholds
                are not in range (0,1].
        """
        if self.gaussian_radius <= 0:
            raise ValueError("gaussian_radius must be positive")
        if not (0 < self.spatial_threshold <= 1):
            raise ValueError("spatial_threshold must be between 0 and 1")
        if not (0 < self.temporal_threshold <= 1):
            raise ValueError("temporal_threshold must be between 0 and 1")


@dataclass
class DetectNewComponents(SupervisedTransformer):
    """Detects new components from residual signals.

    This transformer implements Algorithm 5 (DetectNewComponents) which identifies
    new neural components from the residual buffer after accounting for known
    components. The detection process involves:
    1. Updating and filtering the residual buffer
    2. Finding points of maximum variance
    3. Performing local rank-1 NMF around these points
    4. Validating new components using spatial and temporal correlations
    5. Updating the model when new components are accepted

    The computation follows these key steps:
    - R_buf ← [R_buf[:, 1:l_b-1], y - [A,b][C;f][:, end]]
    - V ← Filter(R_buf - Median(R_buf), GaussianKernel(τ))
    - E ← ∑_i V[:, i]^2
    - [a_new, c_new] = NNMF(R_buf[N_(i_x,i_y), :], 1)

    New components are accepted if they meet correlation thresholds and
    don't duplicate existing components.
    """

    params: DetectNewComponentsParams
    """Configuration parameters for the detection process."""

    footprints_: Footprints = None
    """Updated spatial footprints [A, b]."""

    traces_: Traces = None
    """Updated temporal traces [C; f]."""

    groups_: Overlaps = None
    """Updated component overlaps G as a sparse matrix."""

    residuals_: Residuals = None
    """Updated residual buffer R_buf."""

    pixel_stats_: PixelStats = None
    """Updated pixel-wise sufficient statistics W."""

    component_stats_: ComponentStats = None
    """Updated component-wise sufficient statistics M."""

    is_fitted_: bool = False
    """Indicator whether the transformer has been fitted."""

    def learn_one(
        self,
        frame: xr.DataArray,
        footprints: Footprints,
        traces: Traces,
        residuals: Residuals,
        pixel_stats: PixelStats,
        component_stats: ComponentStats,
        groups: Overlaps,
    ) -> Self:
        """Process current frame to detect new components.

        This method implements the main detection algorithm, processing the
        current frame to identify and validate new components. It maintains
        the residual buffer, performs spatial filtering, and updates the
        model when new components are found.

        Args:
            frame (xr.DataArray): Current data frame y.
                Shape: (height × width)
            footprints (Footprints): Current spatial footprints [A, b].
                Shape: (components × height × width)
            traces (Traces): Current temporal traces [C; f].
                Shape: (components × time)
            residuals (Residuals): Current residual buffer R_buf.
                Shape: (buffer_size × height × width)
            pixel_stats (PixelStats): Sufficient statistics W.
                Shape: (pixels × components)
            component_stats (ComponentStats): Sufficient statistics M.
                Shape: (components × components)
            groups (Overlaps): Current component overlaps G (sparse matrix)

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Steps 1: Update and process residual buffer
        reconstruction = compute_reconstruction(footprints, traces)
        current_residual = frame - reconstruction
        residual_buffer = update_residual_buffer(residuals, current_residual)
        median = np.median(residual_buffer, axis=0)
        residual_buffer = residual_buffer - median

        # Step 2: Apply spatial filtering
        filtered_residual = gaussian_filter(
            residual_buffer, sigma=self.params.gaussian_radius, mode="reflect"
        )

        # Step 3: Compute energy values
        energy = np.sum(filtered_residual**2, axis=0)

        # Step 4: Begin repeat loop for new component detection
        repeat = True
        while repeat:
            # Step 5: Find point of maximum variance
            ix, iy = np.unravel_index(np.argmax(energy), energy.shape)

            # Step 6: Define neighborhood around maximum point
            neighborhood = get_neighborhood(
                ix, iy, self.params.gaussian_radius, residual_buffer.shape[1:]
            )

            # Step 7: Perform local rank-1 NMF in the neighborhood
            a_new, c_new = local_rank1_nmf(
                residual_buffer[:, neighborhood[:, 0], neighborhood[:, 1]]
            )

            # Step 8: Compute spatial correlation
            r = np.corrcoef(a_new, np.mean(residual_buffer, axis=0).flatten())[0, 1]

            # Step 9: Check for overlapping components
            overlaps = find_overlapping_components(a_new, footprints)
            if overlaps:
                # Check temporal correlation with overlapping components
                t_start = residual_buffer.shape[0] - traces.shape[1]
                for j in overlaps:
                    if (
                        np.corrcoef(c_new, traces[j, t_start:])[0, 1]
                        > self.params.temporal_threshold
                    ):
                        r = 0  # Duplicate detected
                        break

            # Steps 10: Accept or reject new component
            if r > self.params.spatial_threshold:
                # Zero-pad the spatial footprint to match full frame size
                a_new_padded = zero_pad_component(a_new, neighborhood, energy.shape)

                # Update component count
                K = footprints.sizes[self.params.component_axis] + 1

                # Update groups using sparse matrix operations
                new_overlaps = compute_new_overlaps(footprints, a_new_padded)
                groups = update_overlap_groups(groups, new_overlaps)

                # Update footprints and traces
                footprints = xr.concat(
                    [footprints, a_new_padded], dim=self.params.component_axis
                )
                traces = xr.concat(
                    [traces, c_new[np.newaxis, :]], dim=self.params.component_axis
                )

                # Update residual buffer and energy
                residual_buffer -= np.outer(c_new, a_new_padded).reshape(
                    (-1,) + energy.shape
                )
                energy = np.sum(residual_buffer**2, axis=0)

                # Update sufficient statistics (W, M)
                pixel_stats, component_stats = update_sufficient_statistics(
                    pixel_stats,
                    component_stats,
                    residual_buffer,  # Y_buf: buffer of recent residual frames
                    traces,  # [C; f]: current temporal components
                    c_new,  # newly detected temporal component
                )
            else:
                repeat = False

        # Store all updated components
        self.footprints_ = footprints
        self.traces_ = traces
        self.groups_ = groups
        self.residuals_ = xr.DataArray(
            residual_buffer, dims=residuals.dims, coords=residuals.coords
        )
        self.pixel_stats_ = pixel_stats
        self.component_stats_ = component_stats

        self.is_fitted_ = True
        return self

    def transform_one(
        self, _=None
    ) -> Tuple[Footprints, Traces, Overlaps, Residuals, PixelStats, ComponentStats]:
        """Return all updated model components.

        Following Algorithm 5, this method returns all updated components including
        spatial footprints, temporal traces, overlap groups (sparse matrix), residual buffer,
        and sufficient statistics matrices.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            tuple:
                - Footprints: Updated spatial footprints [A, b]
                - Traces: Updated temporal traces [C; f]
                - Overlaps: Updated component overlap groups G (sparse matrix)
                - Residuals: Updated residual buffer R_buf
                - PixelStats: Updated pixel-wise sufficient statistics W
                - ComponentStats: Updated component-wise sufficient statistics M

        Raises:
            NotFittedError: If the transformer hasn't been fitted yet.
        """
        if not self.is_fitted_:
            raise NotFittedError

        return (
            self.footprints_,
            self.traces_,
            self.groups_,
            self.residuals_,
            self.pixel_stats_,
            self.component_stats_,
        )


def compute_reconstruction(footprints: Footprints, traces: Traces) -> np.ndarray:
    """Compute reconstruction from current components.

    Args:
        footprints: Spatial footprints [A, b]
        traces: Temporal traces [C; f]

    Returns:
        Reconstructed frame from current components
    """
    return footprints.values @ traces.values


def update_residual_buffer(
    buffer: xr.DataArray, new_residual: xr.DataArray
) -> np.ndarray:
    """Update residual buffer with new frame.

    Args:
        buffer: Current residual buffer
        new_residual: New residual frame to add

    Returns:
        Updated residual buffer with the oldest frame removed and new frame added
    """
    # Roll buffer back one position and add new frame at the end
    buffer_values = buffer.values
    buffer_values = np.roll(buffer_values, -1, axis=0)
    buffer_values[-1] = new_residual.values
    return buffer_values


def get_neighborhood(
    ix: int, iy: int, radius: float, shape: Tuple[int, int]
) -> np.ndarray:
    """Define a neighborhood around a point within given radius.

    Args:
        ix, iy: Center point coordinates
        radius: Neighborhood radius
        shape: Shape of the full frame

    Returns:
        Array of coordinates within the neighborhood
    """
    y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    mask = x * x + y * y <= radius * radius

    # Get valid coordinates within frame bounds
    coords = np.where(mask)
    y_coords = np.clip(coords[0] + ix, 0, shape[0] - 1)
    x_coords = np.clip(coords[1] + iy, 0, shape[1] - 1)

    return np.column_stack([y_coords, x_coords])


def local_rank1_nmf(data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Perform rank-1 NMF on local data.

    Args:
        data: Local spatiotemporal data for NMF

    Returns:
        Spatial and temporal components (a_new, c_new)
    """
    # Initialize with random non-negative values
    h = np.random.rand(data.shape[0])  # temporal
    w = np.random.rand(data.shape[1])  # spatial

    # Simple multiplicative update rules for rank-1 NMF
    for _ in range(10):  # Fixed number of iterations
        h = h * (data.T @ w) / (w.sum() * h)
        w = w * (data @ h) / (h.sum() * w)

        # Normalize
        h_norm = np.linalg.norm(h)
        h = h / h_norm
        w = w * h_norm

    return w, h


def find_overlapping_components(a_new: np.ndarray, footprints: Footprints) -> List[int]:
    """Find components that overlap with the new component.

    Args:
        a_new: New spatial component
        footprints: Existing component footprints

    Returns:
        List of indices of overlapping components
    """
    overlaps = []
    for i in range(footprints.shape[0]):
        if np.sum(a_new * footprints[i].values.flatten()) > 0:
            overlaps.append(i)
    return overlaps


def zero_pad_component(
    a_new: np.ndarray, neighborhood: np.ndarray, full_shape: Tuple[int, int]
) -> np.ndarray:
    """Zero-pad local component to match full frame dimensions.

    Args:
        a_new: Local spatial component
        neighborhood: Neighborhood coordinates
        full_shape: Full frame shape

    Returns:
        Zero-padded spatial component
    """
    padded = np.zeros(full_shape)
    padded[neighborhood[:, 0], neighborhood[:, 1]] = a_new
    return padded


def update_sufficient_statistics(
    pixel_stats: PixelStats,
    component_stats: ComponentStats,
    frame_buffer: xr.DataArray,
    traces_buffer: xr.DataArray,
    c_new: np.ndarray,
) -> Tuple[PixelStats, ComponentStats]:
    """Update sufficient statistics W and M with new component.

    Implements the update equations:
    W_t = [W_t, (1/t)Y_buf c_new^T]
    M_t = (1/t)[tM_t, C_buf c_new; c_new^T C_buf^T, ||c_new||^2]

    where Y_buf and C_buf are the matrices [Y; [C; f]] restricted to
    the last N frames in the buffer.

    Args:
        pixel_stats: Current pixel statistics W_t
        component_stats: Current component statistics M_t
        frame_buffer: Buffer of recent frames Y_buf
        traces_buffer: Buffer of recent temporal components [C; f]
        c_new: New temporal component c_new

    Returns:
        Updated pixel and component statistics (W_t, M_t)
    """
    # Get buffer size and current timestep
    buffer_size = frame_buffer.sizes[frame_buffer.dims[0]]
    t = traces_buffer.sizes[traces_buffer.dims[1]]  # total number of timepoints

    # Get the last N frames from buffers
    Y_buf = frame_buffer.values.reshape(-1, buffer_size)  # pixels × frames
    C_buf = traces_buffer.values[:, -buffer_size:]  # components × frames

    # Update pixel statistics W_t
    # W_t = [W_t, (1/t)Y_buf c_new^T]
    W_new_col = (1 / t) * Y_buf @ c_new  # using matrix multiplication for Y_buf c_new^T
    W_update = np.column_stack([pixel_stats.values, W_new_col])

    # Update component statistics M_t
    # M_t = (1/t)[tM_t, C_buf c_new; c_new^T C_buf^T, ||c_new||^2]
    M_current = component_stats.values
    C_buf_c_new = C_buf @ c_new  # C_buf c_new using matrix multiplication
    c_new_norm = np.dot(c_new, c_new)  # ||c_new||^2

    # Construct new M matrix
    M_top = np.column_stack([t * M_current, C_buf_c_new])
    M_bottom = np.append(C_buf_c_new, c_new_norm)
    M_update = (1 / t) * np.vstack([M_top, M_bottom])

    # Create updated xarray DataArrays with expanded dimensions
    new_component_coord = {
        pixel_stats.dims[1]: np.append(
            pixel_stats.coords[pixel_stats.dims[1]],
            pixel_stats.coords[pixel_stats.dims[1]][-1] + 1,
        )
    }

    updated_pixel_stats = xr.DataArray(
        W_update,
        dims=pixel_stats.dims,
        coords={**pixel_stats.coords, **new_component_coord},
    )

    updated_component_stats = xr.DataArray(
        M_update,
        dims=component_stats.dims,
        coords={**component_stats.coords, **new_component_coord},
    )

    return updated_pixel_stats, updated_component_stats


def compute_new_overlaps(footprints: Footprints, a_new: np.ndarray) -> np.ndarray:
    """Compute overlap between new component and existing components.

    Args:
        footprints: Existing component footprints
        a_new: New component spatial footprint

    Returns:
        Array indicating which components overlap with the new one
    """
    overlaps = np.zeros(footprints.shape[0] + 1, dtype=int)
    for i in range(footprints.shape[0]):
        if np.sum(a_new * footprints[i].values.flatten()) > 0:
            overlaps[i] = 1
            overlaps[-1] = 1  # Mark new component as overlapping
    return overlaps


def update_overlap_groups(groups: Overlaps, new_overlaps: np.ndarray) -> Overlaps:
    """Update overlap groups sparse matrix with new component.

    Args:
        groups: Current overlap groups sparse matrix
        new_overlaps: Overlap information for new component

    Returns:
        Updated sparse matrix including new component
    """
    # Convert to dense temporarily for update
    current_matrix = groups.data.todense()
    n = current_matrix.shape[0]

    # Create new expanded matrix
    new_matrix = np.zeros((n + 1, n + 1), dtype=int)
    new_matrix[:n, :n] = current_matrix

    # Add new component overlaps
    new_matrix[n, :] = new_overlaps
    new_matrix[:, n] = new_overlaps

    # Convert back to sparse and wrap in xarray
    return xr.DataArray(sparse.COO(new_matrix), dims=groups.dims, coords=groups.coords)
