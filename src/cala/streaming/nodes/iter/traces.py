from dataclasses import dataclass, field
from typing import Self

import numpy as np
import xarray as xr
from river.base import SupervisedTransformer
from scipy.sparse.csgraph import connected_components
from sklearn.exceptions import NotFittedError

from cala.streaming.core import Axis, Parameters
from cala.streaming.stores.common import Footprints, Traces
from cala.streaming.stores.odl import Overlaps


@dataclass
class TracesUpdaterParams(Parameters, Axis):
    """Parameters for temporal trace updates.

    This class defines the configuration parameters needed for updating temporal
    traces of components, including axis names and convergence criteria.
    """

    tolerance: float = 1e-3
    """Convergence tolerance level (ε) for the iterative update process."""

    def validate(self) -> None:
        """Validate parameter configurations.

        Raises:
            ValueError: If tolerance is not positive.
        """
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


@dataclass
class TracesUpdater(SupervisedTransformer):
    """Updates temporal traces using an iterative block coordinate descent approach.

    This transformer implements Algorithm 4 (UpdateTraces) which updates temporal
    components using spatial footprints and current frame data. The update process
    uses block coordinate descent with guaranteed convergence under non-negativity
    constraints.

    The update follows the iterative formula:
    c[G_i] = max(c[G_i] + (u[G_i] - V[G_i,:]c)/v[G_i], 0)
    where:
    - c is the temporal traces vector
    - G_i represents component groups
    - u = Ã^T y (projection of current frame)
    - V = Ã^T Ã (gram matrix of spatial components)
    - v = diag{V} (diagonal elements for normalization)
    """

    params: TracesUpdaterParams
    """Configuration parameters for the update process."""

    traces_: xr.DataArray = field(init=False, repr=False)
    """Updated temporal traces for all components."""

    is_fitted_: bool = False
    """Indicator whether the transformer has been fitted."""

    def learn_one(
        self,
        footprints: Footprints,
        frame: xr.DataArray,
        traces: Traces,
        overlaps: Overlaps,
    ) -> Self:
        """Update temporal traces using current spatial footprints and frame data.

        This method implements the block coordinate descent update of temporal
        traces. It processes components based on their overlap relationships,
        ensuring that overlapping components are updated together for proper
        convergence.

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

        Returns:
            Self: The transformer instance for method chaining.
        """
        # Prepare inputs for the update algorithm
        A = footprints.stack({"pixels": self.params.spatial_axes})
        y = frame.stack({"pixels": self.params.spatial_axes})
        c = xr.DataArray(
            traces.isel({self.params.frames_axis: [-1]}),
            coords={
                **traces.coords[Axis.component_axis].coords,
                **{k: (Axis.frames_axis, [v.item()]) for k, v in frame.coords.items()},
            },
        )

        _, labels = connected_components(csgraph=overlaps.data, directed=False, return_labels=True)
        clusters = [np.where(labels == label)[0] for label in np.unique(labels)]

        # Run the update algorithm
        updated_traces = self.update_traces(A, y, c.copy(), clusters, self.params.tolerance)

        # store result with proper coordinates
        self.traces_ = updated_traces

        self.is_fitted_ = True
        return self

    def transform_one(self, _: None = None) -> Traces:
        """Transform the updated traces into the expected format.

        This method wraps the updated temporal traces in a Traces object
        for consistent typing in the pipeline.

        Args:
            _: Unused parameter maintained for API compatibility.

        Returns:
            TraceStore: Wrapped updated temporal traces.

        Raises:
            NotFittedError: If the transformer hasn't been fitted yet.
        """
        if not self.is_fitted_:
            raise NotFittedError

        return self.traces_

    def update_traces(
        self,
        A: xr.DataArray,
        y: xr.DataArray,
        c: xr.DataArray,
        clusters: list[np.ndarray],
        eps: float,
    ) -> xr.DataArray:
        """Implementation of the temporal traces update algorithm.

        This function implements the core update logic of Algorithm 4 (UpdateTraces).
        It uses block coordinate descent to update temporal traces for overlapping
        components together while maintaining non-negativity constraints.

        Args:
            A (xr.DataArray): Spatial footprints matrix [A, b].
                Shape: (components × pixels)
            y (xr.DataArray): Current data frame.
                Shape: (pixels,)
            c (xr.DataArray): Last value of temporal traces. (just used for shape)
                Shape: (components,)
            clusters (list[np.ndarray]): list of groups that each contain component indices that have overlapping footprints.
            eps (float): Tolerance level for convergence checking.

        Returns:
            xr.DataArray: Updated temporal traces satisfying non-negativity constraints.
                Shape: (components,)
        """
        # Step 1: Compute projection of current frame
        u = A @ y

        # Step 2: Compute gram matrix of spatial components
        V = A @ A.rename({self.params.component_axis: f"{self.params.component_axis}'"})

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
                numerator = u.isel({self.params.component_axis: cluster}) - (
                    V.isel({f"{self.params.component_axis}'": cluster}) @ c
                ).rename({f"{self.params.component_axis}'": self.params.component_axis})

                c.loc[{self.params.component_axis: cluster}] = np.maximum(
                    c.isel({self.params.component_axis: cluster})
                    + numerator / np.array([V_diag[cluster]]).T,
                    0,
                )

        return c
