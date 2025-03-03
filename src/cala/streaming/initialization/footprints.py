from dataclasses import dataclass, field
from typing import Self, Tuple, List

import cv2
import numpy as np
import xarray as xr
from river.base import Transformer
from skimage.segmentation import watershed

from cala.streaming.core import Parameters
from cala.streaming.initialization.manager_interface import (
    manager_interface,
    InitializerType,
    FootprintsInitializationResult,
)


@dataclass
class FootprintsInitializerParams(Parameters):
    """Parameters for footprints initialization methods"""

    component_axis: str = "components"
    """Axis for components"""

    threshold_factor: float = 0.2
    """Factor for thresholding distance transform"""
    kernel_size: int = 3
    """Size of kernel for dilation"""
    distance_metric: int = cv2.DIST_L2
    """Distance metric for transform"""
    distance_mask_size: int = 5
    """Mask size for distance transform"""

    def validate(self) -> None:
        if any(
            [
                self.threshold_factor <= 0,
                self.kernel_size <= 0,
                self.distance_mask_size <= 0,
            ]
        ):
            raise ValueError(
                f"Parameters threshold_factor, kernel_size, and distance_mask_size must have positive values."
            )


@manager_interface(InitializerType.FOOTPRINTS)
@dataclass
class FootprintsInitializer(Transformer):
    """Footprints component initialization methods."""

    params: FootprintsInitializerParams
    """Parameters for footprints initialization"""
    spatial_axes: tuple = field(init=False)
    """Spatial axes for footprints"""
    num_markers_: int = field(init=False)
    """Number of markers"""
    markers_: np.ndarray = field(init=False)
    """Markers"""

    result: FootprintsInitializationResult = field(
        default_factory=FootprintsInitializationResult
    )
    """Result from footprints initialization"""

    def learn_one(self, frame: xr.DataArray) -> Self:
        """Learn footprints from a frame."""
        # Get spatial axes
        self.spatial_axes = frame.dims
        # Compute markers
        self.markers_ = self._compute_markers(frame)
        # Extract components
        background, neurons = self._extract_components(self.markers_, frame)

        # Store results
        self.result.background = xr.DataArray(
            background,
            dims=(self.params.component_axis, *self.spatial_axes),
            coords={
                self.params.component_axis: range(len(background)),
                **{axis: frame.coords[axis] for axis in self.spatial_axes},
            },
        )
        self.result.neurons = xr.DataArray(
            neurons,
            dims=(self.params.component_axis, *self.spatial_axes),
            coords={
                self.params.component_axis: range(len(neurons)),
                **{axis: frame.coords[axis] for axis in self.spatial_axes},
            },
        )

        return self

    def transform_one(self, _=None) -> FootprintsInitializationResult:
        """Return initialization result."""

        return self.result

    def _compute_markers(self, frame: xr.DataArray) -> np.ndarray:
        """Compute markers for watershed algorithm."""
        # Convert frame to uint8 before thresholding
        frame_norm = (frame - frame.min()) * (255.0 / (frame.max() - frame.min()))
        frame_uint8 = frame_norm.astype(np.uint8)
        _, binary = cv2.threshold(
            frame_uint8.values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        # Sure background area (by dilating the foreground)
        kernel = np.ones((self.params.kernel_size, self.params.kernel_size), np.uint8)
        sure_background = cv2.dilate(binary, kernel, iterations=1)

        # Compute distance transform of the foreground
        distance = cv2.distanceTransform(
            binary, self.params.distance_metric, self.params.distance_mask_size
        )

        # Threshold the distance transform to get sure foreground
        _, sure_foreground = cv2.threshold(
            distance, self.params.threshold_factor * distance.max(), 255, 0
        )
        sure_foreground = sure_foreground.astype(np.uint8)

        # Identify unknown region
        unknown = cv2.subtract(
            sure_background.astype(np.float32), sure_foreground.astype(np.float32)
        ).astype(np.uint8)

        # Label sure foreground with connected components
        self.num_markers_, markers = cv2.connectedComponents(sure_foreground)

        # Increment labels so background is not 0 but 1
        markers = markers + 1
        # Mark unknown region as 0
        markers[unknown == 255] = 0

        # Call watershed
        return watershed(frame_uint8.values, markers)

    def _extract_components(
        self, markers: np.ndarray, frame: xr.DataArray
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        """Extract background and neurons from markers."""
        background = [(markers == 1) * frame.values]
        neurons = []
        for i in range(2, self.num_markers_ + 1):
            neurons.append((markers == i) * frame.values)
        return background, neurons
