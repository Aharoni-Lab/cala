from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
from cala.streaming.core.components.traits.group import ComponentGroup
from scipy import sparse

from cala.streaming.core.components.background import BackgroundComponent
from cala.streaming.core.components.neuron import NeuralComponent


@dataclass
class Estimates:
    """Stores and manages estimation results for calcium imaging data.

    This class maintains the correlation matrices W and M used for updating spatial
    components, while delegating other functionality to the component types.

    The correlation matrices are:
    - W (pixel_correlations): Correlation between pixels and component activities
    - M (source_correlations): Correlation between component activities
    """

    dimensions: Tuple[int, ...]
    """Dimensions of the imaging field"""

    components: ComponentGroup = field(init=False)
    """Group managing all neural and background components"""

    current_timestamp: int = 0
    """Current timestamp in the processing"""

    shifts: List[Tuple[float, ...]] = field(default_factory=list)
    """Motion stabilization shifts for each frame"""

    pixel_correlations: sparse.csc_matrix = field(init=False)
    """W matrix: correlation between pixels and component activities"""

    source_correlations: np.ndarray = field(init=False)
    """M matrix: correlation between component activities"""

    def __post_init__(self):
        """Initialize component group and correlation matrices"""
        self.components = ComponentGroup(dimensions=self.dimensions)
        n_pixels = np.prod(self.dimensions)
        self.pixel_correlations = sparse.csc_matrix((n_pixels, 0))
        self.source_correlations = np.zeros((0, 0))

    def update(self, frame: np.ndarray) -> None:
        """Update components and correlation matrices with new frame data

        Args:
            frame: Current frame data
        """
        flat_frame = frame.ravel()

        # Get current traces for all components
        traces = []
        for component in [
            *self.components.neural_components.values(),
            *self.components.background_components.values(),
        ]:
            traces.append(component.temporal.fluorescence_trace[-1])
        traces = np.array(traces)

        if traces.size > 0:
            # Update correlation matrices
            t = self.current_timestamp + 1

            # Update W: pixel-component correlations
            outer_product = sparse.csc_matrix(np.outer(flat_frame, traces))
            if t == 1:
                self.pixel_correlations = outer_product
            else:
                weighted_old = sparse.csc_matrix(
                    ((t - 1) / t) * self.pixel_correlations
                )
                weighted_new = sparse.csc_matrix((1 / t) * outer_product)
                self.pixel_correlations = weighted_old + weighted_new

            # Update M: component-component correlations
            if t == 1:
                self.source_correlations = np.outer(traces, traces)
            else:
                self.source_correlations = ((t - 1) / t) * self.source_correlations + (
                    1 / t
                ) * np.outer(traces, traces)

            # Update spatial footprints using W and M
            self._update_spatial_footprints()

        # Update components with new frame
        for component in [
            *self.components.neural_components.values(),
            *self.components.background_components.values(),
        ]:
            component.update(frame)

        self.current_timestamp += 1

    def _update_spatial_footprints(self, max_iter: int = 1) -> None:
        """Update spatial footprints using correlation matrices

        Implements Algorithm 6 (UpdateShapes) from the paper.

        Args:
            max_iter: Maximum number of iterations for the update
        """
        # Get all components in order
        components = [
            *self.components.neural_components.values(),
            *self.components.background_components.values(),
        ]

        if not components:
            return

        # Stack current spatial footprints
        A = sparse.hstack([comp.spatial.footprint for comp in components]).tocsc()

        # Iterate updates
        for _ in range(max_iter):
            for i, component in enumerate(components):
                # Find pixels where component i can be non-zero
                p = component.spatial.footprint.nonzero()[1]
                if len(p) == 0:
                    continue

                # Update rule: A[p,i] = max(A[p,i] + (W[p,i] - A[p,:]M[:,i])/M[i,i], 0)
                interference = A.dot(self.source_correlations[:, i])
                update = (
                    self.pixel_correlations[:, i].toarray().ravel() - interference
                ) / (self.source_correlations[i, i] + 1e-6)

                # Apply update to footprint
                new_footprint = component.spatial.footprint.toarray()
                new_footprint[0, p] = np.maximum(new_footprint[0, p] + update[p], 0)
                component.spatial.footprint = sparse.csc_matrix(new_footprint)

    def add_component(self, component: NeuralComponent | BackgroundComponent) -> None:
        """Add a new component to the estimates

        Args:
            component: Neural or background component to add
        """
        n_comps = len(self.components.neural_components) + len(
            self.components.background_components
        )

        # Extend correlation matrices
        self.pixel_correlations = sparse.hstack(
            [self.pixel_correlations, sparse.csc_matrix((np.prod(self.dimensions), 1))]
        )

        self.source_correlations = np.pad(
            self.source_correlations, ((0, 1), (0, 1)), mode="constant"
        )

        self.components.add_component(component)

    def get_residuals(self, frame: np.ndarray) -> np.ndarray:
        """Compute residuals between frame and current estimates

        Args:
            frame: Current frame to compute residuals for

        Returns:
            Residual frame after subtracting all components
        """
        reconstruction = np.zeros_like(frame)

        # Get reconstruction from all components
        for component in [
            *self.components.neural_components.values(),
            *self.components.background_components.values(),
        ]:
            reconstruction += (
                component.spatial.footprint.toarray().reshape(self.dimensions)
                * component.temporal.fluorescence_trace[-1]
            )

        return frame - reconstruction
