from dataclasses import dataclass, field
from typing import Self

import cv2
import numpy as np
from river.base import SupervisedTransformer
from skimage.segmentation import watershed

from cala.streaming.core import Parameters, Estimates


@dataclass
class SpatialInitializerParams(Parameters):
    """Parameters for spatial initialization methods"""

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


@dataclass
class SpatialInitializer(SupervisedTransformer):
    """Abstract base class for spatial component initialization methods."""

    params: SpatialInitializerParams
    num_markers_: int = field(init=False)
    markers_: np.ndarray = field(init=False)
    blobs_: list[np.ndarray] = field(init=False)

    def learn_one(self, estimates: Estimates, frame: np.ndarray) -> Self:
        # Convert frame to uint8 before thresholding
        frame_norm = (frame - frame.min()) * (255.0 / (frame.max() - frame.min()))
        frame_uint8 = frame_norm.astype(np.uint8)
        _, binary = cv2.threshold(
            frame_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
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
        self.markers_ = watershed(frame_uint8, markers)

        self.blobs_ = []
        for blob_idx in range(2, self.num_markers_ + 1):
            blob_mask = self.markers_ == blob_idx
            blob = blob_mask * frame
            self.blobs_.append(blob)

        return self

    def transform_one(self, estimates: Estimates) -> Estimates:
        estimates.spatial_footprints = np.array(self.blobs_[1:])
        estimates.background_footprints = np.array(self.blobs_[0])

        return estimates
