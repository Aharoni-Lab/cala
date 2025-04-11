from dataclasses import dataclass, field
from typing import Self
from uuid import uuid4

import cv2
import numpy as np
import xarray as xr
from river.base import Transformer
from skimage.segmentation import watershed
from sklearn.exceptions import NotFittedError

from cala.streaming.core import Axis, Component, Parameters
from cala.streaming.stores.common import Footprints


@dataclass
class FootprintsInitializerParams(Parameters, Axis):
    """Parameters for footprints initialization methods"""

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
                "Parameters threshold_factor, kernel_size, and distance_mask_size must have positive values."
            )


@dataclass
class FootprintsInitializer(Transformer):
    """Footprints component initialization methods."""

    params: FootprintsInitializerParams
    """Parameters for footprints initialization"""
    num_markers_: int = field(init=False)
    """Number of markers"""
    markers_: np.ndarray = field(init=False)
    """Markers"""
    footprints_: xr.DataArray = field(init=False)

    is_fitted_: bool = False

    def learn_one(self, frame: xr.DataArray) -> Self:
        """Learn footprints from a frame."""
        # Compute markers
        self.markers_ = self._compute_markers(frame)
        # Extract components
        background, neurons = self._extract_components(self.markers_, frame)

        self.footprints_ = xr.DataArray(
            background + neurons,
            dims=(self.params.component_axis, *frame.dims),
            coords={
                self.params.id_coordinates: (
                    self.params.component_axis,
                    [uuid4().hex for _ in range(len(background) + len(neurons))],
                ),
                self.params.type_coordinates: (
                    self.params.component_axis,
                    [Component.BACKGROUND.value] * len(background)
                    + [Component.NEURON.value] * len(neurons),
                ),
            },
        )

        self.is_fitted_ = True
        return self

    def transform_one(self, _: None = None) -> Footprints:
        """Return initialization result."""
        if not self.is_fitted_:
            raise NotFittedError

        return self.footprints_

    def _compute_markers(self, frame: xr.DataArray) -> np.ndarray:
        """Compute markers for watershed algorithm."""
        # Convert frame to uint8 before thresholding
        frame_norm = (frame - frame.min()) * (255.0 / (frame.max() - frame.min()))
        frame_uint8 = frame_norm.astype(np.uint8)
        _, binary = cv2.threshold(frame_uint8.values, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

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
    ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        """Extract background and neurons from markers."""
        background = [(markers == 1) * frame.values]
        neurons = []
        for i in range(2, self.num_markers_ + 1):
            neurons.append((markers == i) * frame.values)
        return background, neurons
