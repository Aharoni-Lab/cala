from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cala.streaming.components import (
    SpatialInitializer,
    SpatialInitializerParams,
    TemporalInitializer,
    TemporalInitializerParams,
)
from cala.streaming.core import Estimates
from tests.fixtures import stabilized_video


class TestStreamingTemporalInitializer:
    @pytest.fixture
    def spatial_parameters(self):
        return SpatialInitializerParams()

    @pytest.fixture
    def spatial_initializer(self, spatial_parameters):
        return SpatialInitializer(params=spatial_parameters)

    @pytest.fixture
    def temporal_parameters(self):
        return TemporalInitializerParams()

    @pytest.fixture
    def temporal_initializer(self, temporal_parameters):
        return TemporalInitializer(params=temporal_parameters)

    @pytest.fixture
    def spatial_estimates(self, spatial_initializer, stabilized_video):
        video, _, _ = stabilized_video
        frame_dimensions = tuple(video.sizes[d] for d in ["width", "height"])
        default_estimates = Estimates(frame_dimensions)

        estimates = spatial_initializer.learn_one(
            estimates=default_estimates,
            frame=video.values[0],
        ).transform_one(default_estimates)

        return estimates

    def test_first_n_frames(
        self, stabilized_video, temporal_initializer, spatial_estimates
    ):
        video, _, _ = stabilized_video
        temporal_estimates = temporal_initializer.learn_one(
            spatial_estimates, video[:3]
        ).transform_one(spatial_estimates)

        assert (
            spatial_estimates.spatial_footprints.shape[0]
            == temporal_estimates.temporal_traces.shape[0]
        )

    def test_reconstruction_comparison(
        self, stabilized_video, temporal_initializer, spatial_estimates
    ):
        """Test that reconstructed frames from spatial footprints and temporal traces match original frames."""
        video, _, _ = stabilized_video
        temporal_estimates = temporal_initializer.learn_one(
            spatial_estimates, video[:3]
        ).transform_one(spatial_estimates)

        # Get original first 3 frames
        original_frames = video[:3].values

        # Reconstruct frames using spatial footprints and temporal traces
        spatial_footprints = temporal_estimates.spatial_footprints
        temporal_traces = temporal_estimates.temporal_traces

        # Reshape spatial footprints to match frame dimensions
        frame_shape = (video.sizes["height"], video.sizes["width"])
        reshaped_footprints = spatial_footprints.reshape(-1, *frame_shape)

        # Initialize reconstructed frames array
        reconstructed_frames = np.zeros_like(original_frames)

        # For each frame, multiply spatial footprints with corresponding temporal trace values
        for frame_idx in range(3):
            for comp_idx in range(len(spatial_footprints)):
                reconstructed_frames[frame_idx] += (
                    reshaped_footprints[comp_idx] * temporal_traces[comp_idx, frame_idx]
                )

        # Create a figure to display original and reconstructed frames side by side
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle("Original vs Reconstructed Frames Comparison")

        # Plot original frames on top row
        for i in range(3):
            im = axes[0, i].imshow(original_frames[i], cmap="gray")
            axes[0, i].set_title(f"Original Frame {i + 1}")
            axes[0, i].axis("off")
            plt.colorbar(im, ax=axes[0, i])

        # Plot reconstructed frames on bottom row
        for i in range(3):
            im = axes[1, i].imshow(reconstructed_frames[i], cmap="gray")
            axes[1, i].set_title(f"Reconstructed Frame {i + 1}")
            axes[1, i].axis("off")
            plt.colorbar(im, ax=axes[1, i])

        plt.tight_layout()

        # Create artifacts directory if it doesn't exist
        artifact_dir = Path(__file__).parent.parent.parent.parent / "artifacts"
        artifact_dir.mkdir(exist_ok=True)

        # Save the figure
        plt.savefig(artifact_dir / "reconstruction_comparison.png")
        plt.close()
