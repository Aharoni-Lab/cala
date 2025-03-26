from dataclasses import dataclass, field
from typing import Self, Tuple
from uuid import uuid4

import sparse
import xarray as xr
from river.base import SupervisedTransformer
from scipy.ndimage import gaussian_filter
from sklearn.decomposition import NMF

from cala.streaming.composer import Frame
from cala.streaming.core import Parameters, Component
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
            a_new, c_new = self._local_nmf(neighborhood, self.frame_.array)

            # Update residuals and energy
            new_component = a_new * c_new
            self.residuals_ = self.residuals_ - new_component
            V = V - (a_new**2) * (c_new**2).sum()

            # Validate new component
            if not self._validate_component(a_new, c_new, traces, footprints):
                valid = False
                continue

            self.new_footprints_.append(a_new)
            self.new_traces_.append(c_new)

        new_ids = [uuid4() for _ in self.new_footprints_]
        new_types = [Component.NEURON for _ in self.new_footprints_]

        self.new_footprints_ = xr.concat(
            self.new_footprints_, dim=self.params.component_axis
        ).assign_coords(
            {
                self.params.id_coordinates: (self.params.component_axis, new_ids),
                self.params.type_coordinates: (self.params.component_axis, new_types),
            }
        )
        self.new_traces_ = xr.concat(
            self.new_traces_, dim=self.params.component_axis
        ).assign_coords(
            {
                self.params.id_coordinates: (self.params.component_axis, new_ids),
                self.params.type_coordinates: (self.params.component_axis, new_types),
            }
        )
        self.is_fitted_ = True
        return self

    def transform_one(
        self,
        footprints: Footprints,
        traces: Traces,
        residuals: Residuals,
        pixel_stats: PixelStats,
        component_stats: ComponentStats,
        overlaps: Overlaps,
    ) -> Tuple[Footprints, Traces, Residuals, PixelStats, ComponentStats, Overlaps]:
        """

        Args:
            pixel_stats (PixelStats): Sufficient statistics W.
                Shape: (width x height × components)
            component_stats (ComponentStats): Sufficient statistics M.
                Shape: (components × components')
            overlaps (Overlaps): Current component overlaps G (sparse matrix)
                Shape: (components × components'):

        Returns:
            Tuple[Footprints, Traces, Residuals, PixelStats, ComponentStats, Overlaps]:
                - New footprints
                - New traces
                - New residuals
                - New pixel statistics
                - New component statistics
                - New overlaps
        """

        # Update statistics and overlaps
        new_pixel_stats_ = self._update_pixel_stats(
            self.frame_, footprints, traces, residuals, pixel_stats, self.new_traces_
        )
        component_stats_ = self._update_component_stats(
            component_stats, traces, self.new_traces_, self.frame_.index
        )
        overlaps_ = self._update_overlaps(footprints, overlaps, self.new_footprints_)

        return (
            self.new_footprints_,
            self.new_traces_,
            self.residuals_,
            new_pixel_stats_,
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
        ix = max_coords[self.params.spatial_axes[0]].values.tolist()
        iy = max_coords[self.params.spatial_axes[1]].values.tolist()

        # Define neighborhood
        radius = int(self.params.gaussian_radius)
        y_slice = slice(
            max(0, iy - radius),
            min(E.sizes["height"], iy + radius + 1),
        )
        x_slice = slice(
            max(0, ix - radius),
            min(E.sizes["width"], ix + radius + 1),
        )

        # ok embed the actual coordinates onto the array
        neighborhood = E.isel(height=y_slice, width=x_slice).assign_coords(
            {
                self.params.spatial_axes[0]: E.coords[self.params.spatial_axes[0]][
                    x_slice
                ],
                self.params.spatial_axes[1]: E.coords[self.params.spatial_axes[1]][
                    y_slice
                ],
            }
        )

        return neighborhood

    def _local_nmf(
        self,
        neighborhood: xr.DataArray,
        frame: xr.DataArray,
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
            # coords={
            #     self.params.frames_axis: neighborhood.coords[self.params.frames_axis]
            # },
        )

        # Create full-frame zero array with proper coordinates
        a_new = xr.zeros_like(frame)

        # Place the NMF result in the correct location
        a_new.loc[{ax: neighborhood.coords[ax] for ax in self.params.spatial_axes}] = (
            xr.DataArray(
                a.squeeze().reshape(
                    tuple(neighborhood.sizes[ax] for ax in self.params.spatial_axes)
                ),
                dims=self.params.spatial_axes,
                coords={ax: neighborhood.coords[ax] for ax in self.params.spatial_axes},
            )
        )

        # # Normalize spatial component
        # a_new = a_new / a_new.sum()

        return a_new, c_new

    def _validate_component(
        self,
        a_new: xr.DataArray,
        c_new: xr.DataArray,
        traces: Traces,
        footprints: Footprints,
    ) -> bool:
        """Validate new component against spatial and temporal criteria."""
        nonzero_y, nonzero_x = a_new.values.nonzero()
        if len(nonzero_y) == 0:
            return False

        # it should look like something from a residual
        r = xr.corr(
            a_new.isel(
                {
                    self.params.spatial_axes[0]: sorted(list(set(nonzero_y))),
                    self.params.spatial_axes[1]: sorted(list(set(nonzero_x))),
                }
            ),
            self.residuals_.mean(dim=self.params.frames_axis).isel(
                {
                    self.params.spatial_axes[0]: sorted(list(set(nonzero_y))),
                    self.params.spatial_axes[1]: sorted(list(set(nonzero_x))),
                }
            ),
        )
        if r <= self.params.spatial_threshold:
            return False

        # Check for duplicates by computing spatial overlap with existing footprints
        overlaps = xr.dot(a_new, footprints, dims=self.params.spatial_axes)
        overlapping_components = (overlaps > 0).assign_coords(
            {
                self.params.id_coordinates: (
                    self.params.component_axis,
                    footprints.coords[self.params.id_coordinates].values,
                )
            }
        )
        overlapping_components = (
            overlapping_components.where(overlapping_components == True, drop=True)
            .coords[self.params.id_coordinates]
            .values
        )

        if overlapping_components.any():
            # For components with spatial overlap, check temporal correlation
            temporal_corr = xr.corr(
                c_new,
                traces.isel(
                    {
                        self.params.frames_axis: slice(
                            -self.residuals_.sizes[self.params.frames_axis], None
                        )
                    }
                )
                .set_xindex(self.params.id_coordinates)
                .sel(
                    {
                        self.params.id_coordinates: overlapping_components,
                    }
                ),
                dim=self.params.frames_axis,
            )
            if (temporal_corr > self.params.temporal_threshold).any():
                return False

        return True

    def _update_pixel_stats(
        self,
        frame: Frame,
        footprints: Footprints,
        traces: Traces,
        residuals: Residuals,
        pixel_stats: PixelStats,
        new_traces: Traces,
    ) -> PixelStats:
        """Update pixel statistics with new components.

        Updates W_t according to the equation:
        W_t = [W_t, (1/t)Y_buf c_new^T]
        where t is the current frame index.

        Args:
            pixel_stats (PixelStats): Current pixel statistics W_t
            frame (Frame): Current frame with index information
            new_footprints (Footprints): Newly detected spatial components
            new_traces (Traces): Newly detected temporal components

        Returns:
            PixelStats: Updated pixel statistics matrix
        """
        if len(new_traces) == 0:
            return pixel_stats

        # Compute scaling factor (1/t)
        frame_idx = frame.index + 1
        scale = 1 / frame_idx

        # traces has to be the same number of frames as residuals
        y_buf = (
            footprints
            @ traces.isel({self.params.frames_axis: slice(-len(residuals), None)})
            + residuals
        ).stack(pixels=self.params.spatial_axes)
        # Reshape frame to match pixel stats dimensions
        # y_buf = frame.array.stack(pixels=self.params.spatial_axes)

        # Compute outer product of frame and new traces
        # (1/t)Y_buf c_new^T
        new_stats = scale * (y_buf @ new_traces).unstack("pixels")

        # Concatenate with existing pixel stats along component axis
        return xr.concat([pixel_stats, new_stats], dim=self.params.component_axis)

    def _update_component_stats(
        self,
        component_stats: ComponentStats,
        traces: Traces,
        new_traces: Traces,
        frame_idx: int,
    ) -> ComponentStats:
        """Update component statistics with new components.

        Updates M_t according to the equation:
        M_t = (1/t) [ tM_t,         C_buf^T c_new  ]
                 [ c_new C_buf^T, ||c_new||^2   ]
        where:
        - t is the current frame index
        - M_t is the existing component statistics
        - C_buf are the traces in the buffer
        - c_new are the new component traces

        Args:
            component_stats (ComponentStats): Current component statistics M_t
            traces (Traces): Current temporal traces in buffer
            new_traces (Traces): Newly detected temporal components
            frame_idx (int): Current frame index

        Returns:
            ComponentStats: Updated component statistics matrix
        """
        if len(new_traces) == 0:
            return component_stats

        # Get current frame index (1-based)
        t = frame_idx + 1

        # Scale existing statistics: (t-1)/t * M_t
        M_scaled = component_stats * ((t - 1) / t)

        # Compute cross-correlation between buffer and new components
        # C_buf^T c_new
        # C_buf probably has to be the same number of frames as c_new
        bottom_left_corr = (
            traces.sel(
                {
                    self.params.frames_axis: slice(
                        -new_traces.sizes[self.params.frames_axis], None
                    )
                }
            )
            @ new_traces.rename(
                {self.params.component_axis: f"{self.params.component_axis}'"}
            )
            / t
        ).assign_coords(traces.coords)

        top_right_corr = xr.DataArray(
            bottom_left_corr.values,
            dims=bottom_left_corr.dims[::-1],
            coords=new_traces.coords,
        )

        # Compute auto-correlation of new components
        # ||c_new||^2
        auto_corr = (
            new_traces
            @ new_traces.rename(
                {self.params.component_axis: f"{self.params.component_axis}'"}
            )
            / t
        ).assign_coords(new_traces.coords)

        # Create the block matrix structure
        # Top block: [M_scaled, cross_corr]
        top_block = xr.concat(
            [M_scaled, top_right_corr], dim=self.params.component_axis
        )

        # Bottom block: [cross_corr.T, auto_corr]
        bottom_block = xr.concat(
            [bottom_left_corr, auto_corr], dim=self.params.component_axis
        )
        # Combine blocks
        return xr.concat(
            [top_block, bottom_block], dim=f"{self.params.component_axis}'"
        )

    def _update_overlaps(
        self,
        footprints: Footprints,
        overlaps: Overlaps,  # xarray with sparse array (N × N binary adjacency matrix)
        new_footprints: Footprints,
    ) -> Overlaps:
        """Update component overlap matrix with new components.

        Updates the binary adjacency matrix that represents component overlaps.
        Matrix element (i,j) is 1 if components i and j overlap spatially, 0 otherwise.

        Args:
            footprints (Footprints): Current spatial footprints [A, b]
            overlaps (Overlaps): Current overlap matrix as sparse array wrapped in xarray
                Shape: (components × components)
            new_footprints (Footprints): Newly detected spatial components

        Returns:
            Overlaps: Updated overlap matrix including new components
        """
        if len(new_footprints) == 0:
            return overlaps

        # Compute spatial overlaps between new and existing components
        old_new_overlap = footprints.dot(
            new_footprints.rename(
                {self.params.component_axis: f"{self.params.component_axis}'"}
            )
        )
        bottom_left_overlap = (
            (old_new_overlap != 0).astype(int).assign_coords(footprints.coords)
        )

        bottom_left_overlap.values = sparse.COO(bottom_left_overlap.values)

        top_right_overlap = xr.DataArray(
            bottom_left_overlap,
            dims=bottom_left_overlap.dims[::-1],
            coords=new_footprints.coords,
        )

        # Compute overlaps between new components themselves
        new_new_overlaps = new_footprints.dot(
            new_footprints.rename(
                {self.params.component_axis: f"{self.params.component_axis}'"}
            )
        )
        new_new_overlaps = (
            (new_new_overlaps != 0).astype(int).assign_coords(new_footprints.coords)
        )

        new_new_overlaps.values = sparse.COO(new_new_overlaps.values)

        # Construct the new overlap matrix by blocks
        # [existing_overlaps    new_overlaps.T    ]
        # [new_overlaps        new_new_overlaps   ]

        # First concatenate horizontally: [existing_overlaps, old_new_overlaps]
        top_block = xr.concat(
            [overlaps, top_right_overlap], dim=self.params.component_axis
        )

        # Then concatenate vertically with [new_overlaps, new_new_overlaps]
        bottom_block = xr.concat(
            [bottom_left_overlap, new_new_overlaps], dim=self.params.component_axis
        )

        # Finally combine top and bottom blocks
        updated_overlaps = xr.concat(
            [top_block, bottom_block], dim=f"{self.params.component_axis}'"
        )

        return updated_overlaps
