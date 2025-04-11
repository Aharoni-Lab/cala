import logging
from dataclasses import dataclass, field
from typing import Self
from uuid import uuid4

import numpy as np
import sparse
import xarray as xr
from river.base import SupervisedTransformer
from scipy.ndimage import gaussian_filter
from skimage.restoration import estimate_sigma
from sklearn.decomposition import NMF
from sklearn.feature_extraction.image import PatchExtractor

from cala.streaming.core import Axis, Component, Parameters
from cala.streaming.stores.common import Footprints, Traces
from cala.streaming.stores.odl import ComponentStats, Overlaps, PixelStats, Residuals

logger = logging.getLogger(__name__)


@dataclass
class DetectorParams(Parameters, Axis):
    """Parameters for new component detection.

    This class defines the configuration parameters needed for detecting new
    components from residual signals, including thresholds for spatial and
    temporal correlations, and filtering parameters for spatial processing.
    """

    num_nmf_residual_frames: int
    """The number of past frames to use for NMF."""

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
class Detector(SupervisedTransformer):
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

    params: DetectorParams
    """Configuration parameters for the detection process."""

    sampler: PatchExtractor = PatchExtractor(patch_size=(20, 20), max_patches=30)

    frame_: xr.DataArray = None

    noise_level_: float = field(init=False)

    cell_radius_: float = field(init=False)

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

    def learn_one(
        self,
        frame: xr.DataArray,
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
        self.cell_radius_ = self._estimate_cell_radius(footprints)
        logger.info(f"Cell Radius: {self.cell_radius_:4f}")

        # Update and process residuals
        self.residuals_ = self._update_residual_buffer(
            frame=frame, footprints=footprints, traces=traces, residuals=residuals
        )

        self.noise_level_ = self._estimate_gaussian_noise(residuals, frame.shape)
        logger.info(f"Noise Level: {self.noise_level_:4f}")

        valid = True
        while valid:
            # Compute deviation from median
            V = self._process_residuals()
            if (V.max() - V.min()) / 2 <= self.noise_level_:  # if fluctuation is noise level
                valid = False
                continue

            # Compute energy (variance)
            E = (V**2).sum(dim=self.params.frames_axis)

            # Find and analyze neighborhood of maximum variance
            neighborhood = self._get_max_variance_neighborhood(E)
            a_new, c_new = self._local_nmf(neighborhood=neighborhood, frame=self.frame_)

            logger.info(f"New C: {c_new.values}")
            # Validate new component
            if not self._validate_component(
                a_new=a_new, c_new=c_new, traces=traces, footprints=footprints
            ):
                valid = False
                continue

            # Update residuals and energy
            new_component = a_new * c_new
            self.residuals_ = self.residuals_ - new_component
            # V = V - (a_new**2) * (c_new**2).sum()

            self.new_footprints_.append(a_new)
            self.new_traces_.append(c_new)

        if len(self.new_footprints_) == 0:
            self.new_footprints_ = xr.DataArray([])
            self.new_traces_ = xr.DataArray([])
            return self

        new_ids = [uuid4().hex for _ in self.new_footprints_]
        new_types = [Component.NEURON.value for _ in self.new_footprints_]
        new_coords = {
            self.params.id_coordinates: (self.params.component_axis, new_ids),
            self.params.type_coordinates: (self.params.component_axis, new_types),
        }

        self.new_footprints_ = xr.concat(
            self.new_footprints_, dim=self.params.component_axis
        ).assign_coords(new_coords)
        self.new_traces_ = xr.concat(
            self.new_traces_, dim=self.params.component_axis
        ).assign_coords(new_coords)
        self.is_fitted_ = True
        return self

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
            frame=self.frame_,
            og_footprints=footprints,
            new_footprints=self.new_footprints_,
            og_traces=traces,
            new_traces=self.new_traces_,
            residuals=self.residuals_,
            pixel_stats=pixel_stats,
        )
        component_stats_ = self._update_component_stats(
            frame_idx=self.frame_.coords[Axis.frame_coordinates].item(),
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

    def _estimate_cell_radius(self, footprints: Footprints) -> float:
        neuron_footprints = footprints.set_xindex(self.params.type_coordinates).sel(
            {self.params.type_coordinates: Component.NEURON.value}
        )
        if self.params.component_axis not in neuron_footprints.dims:
            neuron_footprints = neuron_footprints.expand_dims(self.params.component_axis)
        avg_footprint = (neuron_footprints != 0).sum() / neuron_footprints.sizes[
            self.params.component_axis
        ]
        return max(float(np.sqrt(avg_footprint)) / 2, 1.0)

    def _estimate_gaussian_noise(self, residuals: Residuals, frame_shape: tuple[int, ...]) -> float:
        self.sampler.patch_size = min(self.sampler.patch_size, frame_shape)
        patches = self.sampler.transform(residuals)
        return float(estimate_sigma(patches))

    def _update_residual_buffer(
        self,
        frame: xr.DataArray,
        footprints: Footprints,
        traces: Traces,
        residuals: Residuals,
    ) -> Residuals:
        """Update residual buffer with new frame."""
        prediction = footprints @ traces.isel({self.params.frames_axis: -1})
        new_residual = frame - prediction
        if len(residuals) >= self.params.num_nmf_residual_frames:
            n_frames_discard = len(residuals) - self.params.num_nmf_residual_frames + 1
            residual_slice = residuals.isel(
                {self.params.frames_axis: slice(n_frames_discard, None)}
            )
        else:
            residual_slice = residuals
        residuals = xr.concat(
            [residual_slice, new_residual],
            dim=self.params.frames_axis,
        )
        return residuals

    def _process_residuals(self) -> xr.DataArray:
        """Process residuals through median subtraction and spatial filtering."""
        # Center residuals: why median and not mean?
        R_med = self.residuals_.median(dim=self.params.frames_axis)
        R_centered = self.residuals_ - R_med

        # Apply spatial filter -- why are we doing this??
        V = xr.apply_ufunc(
            lambda x: gaussian_filter(x, self.params.gaussian_std),
            R_centered,
            input_core_dims=[[*self.params.spatial_axes]],
            output_core_dims=[[*self.params.spatial_axes]],
            vectorize=True,
            dask="allowed",
        )

        return V

    def _get_max_variance_neighborhood(
        self,
        E: xr.DataArray,
    ) -> xr.DataArray:
        """Find neighborhood around point of maximum variance."""
        # Find maximum point
        max_coords = E.argmax(dim=self.params.spatial_axes)

        # Define neighborhood
        radius = int(np.round(self.cell_radius_))
        window = {
            ax: slice(
                max(0, pos.values - radius),
                min(E.sizes[ax], pos.values + radius + 1),
            )
            for ax, pos in max_coords.items()
        }

        # ok embed the actual coordinates onto the array
        neighborhood = self.residuals_.isel(window).assign_coords(
            {ax: E.coords[ax][pos] for ax, pos in window.items()}
        )

        return neighborhood

    def _local_nmf(
        self,
        neighborhood: xr.DataArray,
        frame: xr.DataArray,
    ) -> tuple[xr.DataArray, xr.DataArray]:
        """Perform local rank-1 Non-negative Matrix Factorization.

        Uses scikit-learn's NMF implementation to decompose the neighborhood
        into spatial (a) and temporal (c) components.

        Args:
            neighborhood (xr.DataArray): Local region of residual buffer.
                Shape: (frames × height × width)

        Returns:
            tuple[xr.DataArray, xr.DataArray]:
                - Spatial component a_new (height × width)
                - Temporal component c_new (frames)
        """
        # Reshape neighborhood to 2D matrix (time × space)
        R = neighborhood.stack(space=self.params.spatial_axes).transpose(
            self.params.frames_axis, "space"
        )

        # Apply NMF
        model = NMF(n_components=1, init="random")
        logger.info(f"R Sizes: {R.sizes}")
        # when residual is negative, the error becomes massive...
        c = model.fit_transform(R.clip(0))  # temporal component
        a = model.components_  # spatial component

        # Convert back to xarray with proper dimensions and coordinates
        c_new = xr.DataArray(
            c.squeeze(),
            dims=[self.params.frames_axis],
            coords=self.residuals_.coords[Axis.frames_axis].coords,
        )

        # Create full-frame zero array with proper coordinates
        a_new = xr.DataArray(np.zeros_like(frame.values), dims=frame.dims)

        # Place the NMF result in the correct location
        a_new.loc[{ax: neighborhood.coords[ax] for ax in self.params.spatial_axes}] = xr.DataArray(
            a.squeeze().reshape(tuple(neighborhood.sizes[ax] for ax in self.params.spatial_axes)),
            dims=self.params.spatial_axes,
            coords={ax: neighborhood.coords[ax] for ax in self.params.spatial_axes},
        )

        return a_new, c_new

    def _validate_component(
        self,
        a_new: xr.DataArray,
        c_new: xr.DataArray,
        traces: Traces,
        footprints: Footprints,
    ) -> bool:
        """Validate new component against spatial and temporal criteria."""
        nonzero_ax_to_idx = {
            ax: sorted([int(x) for x in set(idx)])
            for ax, idx in zip(a_new.dims, a_new.values.nonzero())
        }

        if len(list(nonzero_ax_to_idx.values())[0]) == 0:
            return False

        # it should look like something from a residual. paper does not specify this,
        # but i think we should only get correlation from the new footprint perimeter,
        # since otherwise the correlation will get drowned out by the mismatch
        # from where the detected cell is NOT present.
        mean_residual = self.residuals_.mean(dim=self.params.frames_axis)

        a_norm = a_new.isel(nonzero_ax_to_idx) / a_new.sum()
        res_norm = mean_residual.isel(nonzero_ax_to_idx) / mean_residual.sum()
        r_spatial = xr.corr(a_norm, res_norm) if np.abs(a_norm - res_norm).max() > 1e-7 else 1.0

        if r_spatial <= self.params.spatial_threshold:
            return False

        # Check for duplicates by computing spatial overlap with existing footprints
        overlaps = (a_new @ footprints) > 0

        overlapping_components = (
            overlaps.where(overlaps, drop=True).coords[self.params.id_coordinates].values
        )

        if overlapping_components.any():
            relevant_traces = (
                traces.set_xindex(self.params.id_coordinates)
                .sel(
                    {
                        self.params.id_coordinates: overlapping_components,
                    }
                )
                .isel(
                    {
                        self.params.frames_axis: slice(
                            -self.residuals_.sizes[self.params.frames_axis], None
                        )
                    }
                )
            )
            # For components with spatial overlap, check temporal correlation
            temporal_corr = xr.corr(
                c_new,
                relevant_traces,
                dim=self.params.frames_axis,
            )

            if (temporal_corr > self.params.temporal_threshold).any():
                return False

        return True

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
        frame_idx = frame.coords[Axis.frame_coordinates].item() + 1
        scale = 1 / frame_idx

        footprints = xr.concat([og_footprints, new_footprints], dim=self.params.component_axis)
        traces = xr.concat(
            [
                og_traces.isel({self.params.frames_axis: slice(-len(residuals), None)}),
                new_traces,
            ],
            dim=self.params.component_axis,
        )

        # traces has to be the same number of frames as residuals
        y_buf = footprints @ traces + residuals

        # Compute outer product of frame and new traces
        # (1/t)Y_buf c_new^T
        new_stats = scale * (y_buf @ new_traces)

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

        M = component_stats

        # Compute cross-correlation between buffer and new components
        # C_buf^T c_new
        # C_buf probably has to be the same number of frames as c_new
        bottom_left_corr = (
            traces.sel(
                {self.params.frames_axis: slice(-new_traces.sizes[self.params.frames_axis], None)}
            )
            @ new_traces.rename({self.params.component_axis: f"{self.params.component_axis}'"})
            / t
        ).assign_coords(traces.coords[Axis.component_axis].coords)

        top_right_corr = xr.DataArray(
            bottom_left_corr.values,
            dims=bottom_left_corr.dims[::-1],
            coords=new_traces.coords[Axis.component_axis].coords,
        )

        # Compute auto-correlation of new components
        # ||c_new||^2
        auto_corr = (
            new_traces
            @ new_traces.rename({self.params.component_axis: f"{self.params.component_axis}'"})
            / t
        ).assign_coords(new_traces.coords[Axis.component_axis].coords)

        # Create the block matrix structure
        # Top block: [M_scaled, cross_corr]
        top_block = xr.concat([M, top_right_corr], dim=self.params.component_axis)

        # Bottom block: [cross_corr.T, auto_corr]
        bottom_block = xr.concat([bottom_left_corr, auto_corr], dim=self.params.component_axis)
        # Combine blocks
        return xr.concat([top_block, bottom_block], dim=f"{self.params.component_axis}'")

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
            new_footprints.rename({self.params.component_axis: f"{self.params.component_axis}'"})
        )
        bottom_left_overlap = (
            (old_new_overlap != 0)
            .astype(int)
            .assign_coords(
                {
                    self.params.id_coordinates: (
                        self.params.component_axis,
                        footprints.coords[self.params.id_coordinates].values,
                    ),
                    self.params.type_coordinates: (
                        self.params.component_axis,
                        footprints.coords[self.params.type_coordinates].values,
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
            new_footprints.rename({self.params.component_axis: f"{self.params.component_axis}'"})
        )
        new_new_overlaps = (new_new_overlaps != 0).astype(int).assign_coords(new_footprints.coords)

        new_new_overlaps.values = sparse.COO(new_new_overlaps.values)

        # Construct the new overlap matrix by blocks
        # [existing_overlaps    new_overlaps.T    ]
        # [new_overlaps        new_new_overlaps   ]

        # First concatenate horizontally: [existing_overlaps, old_new_overlaps]
        top_block = xr.concat([overlaps, top_right_overlap], dim=self.params.component_axis)

        # Then concatenate vertically with [new_overlaps, new_new_overlaps]
        bottom_block = xr.concat(
            [bottom_left_overlap, new_new_overlaps], dim=self.params.component_axis
        )

        # Finally combine top and bottom blocks
        updated_overlaps = xr.concat(
            [top_block, bottom_block], dim=f"{self.params.component_axis}'"
        )

        return updated_overlaps
