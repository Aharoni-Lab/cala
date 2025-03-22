from dataclasses import dataclass, field
from typing import Self, Tuple

import xarray as xr
from river.base import SupervisedTransformer
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF

from cala.streaming.composer import Frame
from cala.streaming.core import Parameters
from cala.streaming.stores.common import Footprints, Traces
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

    id_coordinates: str = "id_"
    """Name of the coordinate representing component IDs. (attached to the component_axis)"""

    type_coordinates: str = "type_"
    """Name of the coordinate representing component types. (attached to the component_axis)"""

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

    new_footprints_: Footprints = field(default_factory=list)
    """New spatial footprints [A, b]."""

    new_traces_: Traces = field(default_factory=list)
    """New temporal traces [C; f]."""

    overlaps_: Overlaps = None
    """Updated component overlaps G as a sparse matrix."""

    residuals_: Residuals = None
    """Updated residual buffer R_buf."""

    frame_: Frame = None

    is_fitted_: bool = False
    """Indicator whether the transformer has been fitted."""

    def learn_one(
        self,
        frame: Frame,
        footprints: Footprints,
        traces: Traces,
        residuals: Residuals,
        overlaps: Overlaps,
    ) -> Self:
        """Process current frame to detect new components.

        This method implements the main detection algorithm, processing the
        current frame to identify and validate new components. It maintains
        the residual buffer, performs spatial filtering, and updates the
        model when new components are found.

        Args:
            frame (Frame): Current data frame y.
                Shape: (height × width)
            footprints (Footprints): Current spatial footprints [A, b].
                Shape: (components × height × width)
            traces (Traces): Current temporal traces [C; f].
                Shape: (components × time)
            residuals (Residuals): Current residual buffer R_buf.
                Shape: (buffer_size × height × width)
            overlaps (Overlaps): Current component overlaps G (sparse matrix)
                Shape: (components × components')

        Returns:
            Self: The transformer instance for method chaining.
        """

        self.frame_ = frame
        # Update and process residuals
        self._update_residual_buffer(frame.array, footprints, traces, residuals)
        V = self._process_residuals()

        valid = True
        while valid:
            # Compute energy
            E = (V**2).sum(dim=self.params.frames_axis)

            # Find and analyze neighborhood of maximum variance
            neighborhood = self._get_max_variance_neighborhood(E)
            a_new, c_new = self._local_nmf(neighborhood)

            # Update residuals and energy
            new_component = a_new * c_new
            self.residuals_ = self.residuals_ - new_component
            V = V - (a_new**2) * (c_new**2).sum()

            # Validate new component
            if not self._validate_component(a_new, c_new, traces, overlaps):
                valid = False
                continue

            self.new_footprints_.append(a_new)
            self.new_traces_.append(c_new)

        self.is_fitted_ = True
        self.new_footprints_ = xr.concat(
            self.new_footprints_, dim=self.params.component_axis
        )
        self.new_traces_ = xr.concat(self.new_traces_, dim=self.params.component_axis)
        return self

    def transform_one(
        self,
        pixel_stats: PixelStats,
        component_stats: ComponentStats,
        overlaps: Overlaps,
    ) -> Tuple[Footprints, Traces, Residuals, PixelStats, ComponentStats, Overlaps]:
        """

        Args:
            pixel_stats (PixelStats): Sufficient statistics W.
                Shape: (pixels × components)
            component_stats (ComponentStats): Sufficient statistics M.
                Shape: (components × components')
            overlaps (Overlaps): Current component overlaps G (sparse matrix)
                Shape: (components × components'):

        Returns:

        """

        # Update statistics and overlaps
        pixel_stats_, component_stats_ = self._update_sufficient_statistics(
            pixel_stats, component_stats, self.frame_, self.new_traces_
        )
        overlaps_ = self._update_overlaps(overlaps, self.new_footprints_)

        return (
            self.new_footprints_,
            self.new_traces_,
            self.residuals_,
            pixel_stats_,
            component_stats_,
            overlaps_,
        )

    def _update_residual_buffer(
        self,
        frame: xr.DataArray,
        footprints: Footprints,
        traces: Traces,
        residuals: Residuals,
    ) -> None:
        """Update residual buffer with new frame."""
        prediction = (footprints * traces.isel({self.params.frames_axis: -1})).sum(
            dim=self.params.component_axis
        )
        new_residual = frame - prediction
        self.residuals_ = xr.concat(
            [
                residuals.isel({self.params.frames_axis: slice(1, None)}),
                new_residual.expand_dims(self.params.frames_axis),
            ],
            dim=self.params.frames_axis,
        )

    def _process_residuals(self) -> xr.DataArray:
        """Process residuals through median subtraction and spatial filtering."""
        # Center residuals
        R_med = self.residuals_.median(dim=self.params.frames_axis)
        R_centered = self.residuals_ - R_med

        # Apply spatial filter
        V = xr.apply_ufunc(
            lambda x: gaussian_filter(x, self.params.gaussian_radius),
            R_centered,
            input_core_dims=[[*self.params.spatial_axes]],
            output_core_dims=[[*self.params.spatial_axes]],
            vectorize=True,
        )

        return V

    def _get_max_variance_neighborhood(
        self,
        E: xr.DataArray,
    ) -> xr.DataArray:
        """Find neighborhood around point of maximum variance."""
        # Find maximum point
        max_coords = E.argmax(dim=self.params.spatial_axes)
        ix = max_coords[self.params.spatial_axes[0]]
        iy = max_coords[self.params.spatial_axes[1]]

        # Define neighborhood
        radius = self.params.gaussian_radius
        return self.residuals_.sel(
            {
                self.params.spatial_axes[0]: slice(
                    max(0, ix - radius),
                    min(E.sizes[self.params.spatial_axes[0]], ix + radius + 1),
                ),
                self.params.spatial_axes[1]: slice(
                    max(0, iy - radius),
                    min(E.sizes[self.params.spatial_axes[1]], iy + radius + 1),
                ),
            }
        )

    def _validate_component(
        self,
        a_new: xr.DataArray,
        c_new: xr.DataArray,
        traces: Traces,
        overlaps: Overlaps,
    ) -> bool:
        """Validate new component against spatial and temporal criteria."""
        # Check spatial correlation
        r = xr.corr(
            a_new,
            self.residuals_.mean(dim=self.params.frames_axis),
            dim=self.params.spatial_axes,
        )
        if r <= self.params.spatial_threshold:
            return False

        # Check for duplicates
        overlapping = overlaps.sel({self.params.component_axis: a_new > 0})
        if len(overlapping) > 0:
            temporal_corr = xr.corr(
                c_new,
                traces.isel(
                    {
                        self.params.frames_axis: slice(
                            -self.residuals_.sizes[self.params.frames_axis], None
                        )
                    }
                ),
                dim=self.params.frames_axis,
            )
            if (temporal_corr > self.params.temporal_threshold).any():
                return False

        return True

    def _local_nmf(
        self,
        neighborhood: xr.DataArray,
    ) -> Tuple[xr.DataArray, xr.DataArray]:
        """Perform local rank-1 Non-negative Matrix Factorization.

        Uses scikit-learn's NMF implementation to decompose the neighborhood
        into spatial (a) and temporal (c) components.

        Args:
            neighborhood (xr.DataArray): Local region of residual buffer.
                Shape: (frames × height × width)

        Returns:
            Tuple[xr.DataArray, xr.DataArray]:
                - Spatial component a_new (height × width)
                - Temporal component c_new (frames)
        """
        # Reshape neighborhood to 2D matrix (time × space)
        R = neighborhood.stack(space=self.params.spatial_axes).transpose(
            self.params.frames_axis, "space"
        )

        # Apply NMF
        model = NMF(n_components=1, init="random")
        c = model.fit_transform(R)  # temporal component
        a = model.components_  # spatial component

        # Convert back to xarray with proper dimensions and coordinates
        c_new = xr.DataArray(
            c.squeeze(),
            dims=[self.params.frames_axis],
            coords={
                self.params.frames_axis: neighborhood.coords[self.params.frames_axis]
            },
        )

        a_new = xr.DataArray(
            a.squeeze().reshape(
                tuple(neighborhood.sizes[ax] for ax in self.params.spatial_axes)
            ),
            dims=self.params.spatial_axes,
            coords={ax: neighborhood.coords[ax] for ax in self.params.spatial_axes},
        )

        # Normalize spatial component
        a_new = a_new / a_new.sum()

        return a_new, c_new
