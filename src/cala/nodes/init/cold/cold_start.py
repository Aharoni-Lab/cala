import logging
from dataclasses import dataclass, field
from typing import Self

import sparse
import xarray as xr
from river.base import SupervisedTransformer
from sklearn.feature_extraction.image import PatchExtractor

from cala.models import Params
from cala.stores.common import Footprints, Traces
from cala.stores.odl import ComponentStats, Overlaps, PixelStats, Residuals

logger = logging.getLogger(__name__)


@dataclass
class ColdStarterParams(Params):
    """Parameters for new component detection.

    This class defines the configuration parameters needed for detecting new
    components from residual signals, including thresholds for spatial and
    temporal correlations, and filtering parameters for spatial processing.
    """

    num_frames: int
    """The number of past frames to use."""

    cell_radius: int

    gaussian_std: float = 1.0
    """Radius (τ) of Gaussian kernel for spatial filtering."""

    spatial_threshold: float = 0.8
    """Threshold for correlation in space (r_s)."""

    temporal_threshold: float = 0.8
    """Threshold for correlation in time (r_t)."""

    def validate(self) -> None:
        """Validate parameter configurations.

        Raises:
            ValueError: If gaussian_radius is not positive or if thresholds
                are not in range (0,1].
        """
        if self.gaussian_std <= 0:
            raise ValueError("gaussian_radius must be positive")
        if not (0 < self.spatial_threshold <= 1):
            raise ValueError("spatial_threshold must be between 0 and 1")
        if not (0 < self.temporal_threshold <= 1):
            raise ValueError("temporal_threshold must be between 0 and 1")


@dataclass
class ColdStarter(SupervisedTransformer):
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
    - [a_new, c_new] = NMF(R_buf[N_(i_x,i_y), :], 1)

    New components are accepted if they meet correlation thresholds and
    don't duplicate existing components.
    """

    params: ColdStarterParams
    """Configuration parameters for the detection process."""

    sampler: PatchExtractor = PatchExtractor(patch_size=(20, 20), max_patches=30)

    noise_level_: float = field(init=False)

    new_footprints_: Footprints = field(default_factory=list)
    """New spatial footprints [A, b]."""

    new_traces_: Traces = field(default_factory=list)
    """New temporal traces [C; f]."""

    overlaps_: Overlaps = None
    """Updated component overlaps G as a sparse matrix."""

    residuals_: Residuals = None
    """Updated residual buffer R_buf."""

    is_fitted_: bool = False
    """Indicator whether the transformer has been fitted."""

    def learn_one(self, x: xr.DataArray, y: xr.DataArray) -> Self: ...

    def transform_one(
        self,
        footprints: Footprints,
        traces: Traces,
        pixel_stats: PixelStats,
        component_stats: ComponentStats,
        overlaps: Overlaps,
    ) -> tuple[Footprints, Traces, Residuals, PixelStats, ComponentStats, Overlaps]:
        """

        Args:
            pixel_stats (PixelStats): Sufficient statistics W.
                Shape: (width x height × components)
            component_stats (ComponentStats): Sufficient statistics M.
                Shape: (components × components')
            overlaps (Overlaps): Current component overlaps G (sparse matrix)
                Shape: (components × components'):

        Returns:
            tuple[Footprints, Traces, Residuals, PixelStats, ComponentStats, Overlaps]:
                - New footprints
                - New traces
                - New residuals
                - New pixel statistics
                - New component statistics
                - New overlaps
        """

        # Update statistics and overlaps
        pixel_stats_ = self._update_pixel_stats(
            frame=self.residuals_,
            og_footprints=footprints,
            new_footprints=self.new_footprints_,
            og_traces=traces,
            new_traces=self.new_traces_,
            residuals=self.residuals_,
            pixel_stats=pixel_stats,
        )
        component_stats_ = self._update_component_stats(
            frame_idx=self.residuals_.coords[self.params.frame_coord].item(),
            traces=traces,
            new_traces=self.new_traces_,
            component_stats=component_stats,
        )
        overlaps_ = self._update_overlaps(
            footprints=footprints,
            new_footprints=self.new_footprints_,
            overlaps=overlaps,
        )

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
    ) -> Residuals:
        """Update residual buffer with new frame."""
        prediction = footprints @ traces.isel({self.params.frames_dim: -1})
        new_residual = frame - prediction
        if len(residuals) >= self.params.num_nmf_residual_frames:
            n_frames_discard = len(residuals) - self.params.num_nmf_residual_frames + 1
            residual_slice = residuals.isel({self.params.frames_dim: slice(n_frames_discard, None)})
        else:
            residual_slice = residuals
        residuals = xr.concat(
            [residual_slice, new_residual],
            dim=self.params.frames_dim,
        )
        return residuals

    def _update_pixel_stats(
        self,
        frame: xr.DataArray,
        og_footprints: Footprints,
        new_footprints: Footprints,
        og_traces: Traces,
        new_traces: Traces,
        residuals: Residuals,
        pixel_stats: PixelStats,
    ) -> PixelStats:
        """Update pixel statistics with new components.

        Updates W_t according to the equation:
        W_t = [W_t, (1/t)Y_buf c_new^T]
        where t is the current frame index.

        Args:
            pixel_stats (PixelStats): Current pixel statistics W_t
            frame (Frame): Current frame with index information
            new_traces (Traces): Newly detected temporal components

        Returns:
            PixelStats: Updated pixel statistics matrix
        """
        if len(new_traces) == 0:
            return pixel_stats

        # Compute scaling factor (1/t)
        frame_idx = frame.coords[self.params.frame_coord].item() + 1
        scale = 1 / frame_idx

        footprints = xr.concat([og_footprints, new_footprints], dim=self.params.component_dim)
        traces = xr.concat(
            [
                og_traces.isel({self.params.frames_dim: slice(-len(residuals), None)}),
                new_traces,
            ],
            dim=self.params.component_dim,
        )

        # traces has to be the same number of frames as residuals
        y_buf = footprints @ traces + residuals

        # Compute outer product of frame and new traces
        # (1/t)Y_buf c_new^T
        new_stats = scale * (y_buf @ new_traces)

        # Concatenate with existing pixel stats along component axis
        return xr.concat([pixel_stats, new_stats], dim=self.params.component_dim)

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

        M = component_stats

        # Compute cross-correlation between buffer and new components
        # C_buf^T c_new
        # C_buf probably has to be the same number of frames as c_new
        bottom_left_corr = (
            traces.sel(
                {self.params.frames_dim: slice(-new_traces.sizes[self.params.frames_dim], None)}
            )
            @ new_traces.rename({self.params.component_dim: f"{self.params.component_dim}'"})
            / t
        ).assign_coords(traces.coords[self.params.component_dim].coords)

        top_right_corr = xr.DataArray(
            bottom_left_corr.values,
            dims=bottom_left_corr.dims[::-1],
            coords=new_traces.coords[self.params.component_dim].coords,
        )

        # Compute auto-correlation of new components
        # ||c_new||^2
        auto_corr = (
            new_traces
            @ new_traces.rename({self.params.component_dim: f"{self.params.component_dim}'"})
            / t
        ).assign_coords(new_traces.coords[self.params.component_dim].coords)

        # Create the block matrix structure
        # Top block: [M_scaled, cross_corr]
        top_block = xr.concat([M, top_right_corr], dim=self.params.component_dim)

        # Bottom block: [cross_corr.T, auto_corr]
        bottom_block = xr.concat([bottom_left_corr, auto_corr], dim=self.params.component_dim)
        # Combine blocks
        return xr.concat([top_block, bottom_block], dim=f"{self.params.component_dim}'")

    def _update_overlaps(
        self,
        footprints: Footprints,
        new_footprints: Footprints,
        overlaps: Overlaps,  # xarray with sparse array (N × N binary adjacency matrix)
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
            new_footprints.rename({self.params.component_dim: f"{self.params.component_dim}'"})
        )
        bottom_left_overlap = (
            (old_new_overlap != 0)
            .astype(int)
            .assign_coords(
                {
                    self.params.id_coord: (
                        self.params.component_dim,
                        footprints.coords[self.params.id_coord].values,
                    ),
                    self.params.type_coord: (
                        self.params.component_dim,
                        footprints.coords[self.params.type_coord].values,
                    ),
                }
            )
        )

        bottom_left_overlap.values = sparse.COO(bottom_left_overlap.values)

        top_right_overlap = xr.DataArray(
            bottom_left_overlap,
            dims=bottom_left_overlap.dims[::-1],
            coords=new_footprints.coords,
        )

        # Compute overlaps between new components themselves
        new_new_overlaps = new_footprints.dot(
            new_footprints.rename({self.params.component_dim: f"{self.params.component_dim}'"})
        )
        new_new_overlaps = (new_new_overlaps != 0).astype(int).assign_coords(new_footprints.coords)

        new_new_overlaps.values = sparse.COO(new_new_overlaps.values)

        # Construct the new overlap matrix by blocks
        # [existing_overlaps    new_overlaps.T    ]
        # [new_overlaps        new_new_overlaps   ]

        # First concatenate horizontally: [existing_overlaps, old_new_overlaps]
        top_block = xr.concat([overlaps, top_right_overlap], dim=self.params.component_dim)

        # Then concatenate vertically with [new_overlaps, new_new_overlaps]
        bottom_block = xr.concat(
            [bottom_left_overlap, new_new_overlaps], dim=self.params.component_dim
        )

        # Finally combine top and bottom blocks
        updated_overlaps = xr.concat([top_block, bottom_block], dim=f"{self.params.component_dim}'")

        return updated_overlaps
