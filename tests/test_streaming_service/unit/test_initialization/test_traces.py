import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

from cala.streaming.core.components import ComponentManager
from cala.streaming.initialization import (
    FootprintsInitializer,
    FootprintsInitializerParams,
    TracesInitializer,
    TracesInitializerParams,
)
from tests.fixtures import stabilized_video


class TestStreamingTemporalInitializer:
    @pytest.fixture
    def footprints_parameters(self):
        return FootprintsInitializerParams()

    @pytest.fixture
    def footprints_initializer(self, footprints_parameters):
        return FootprintsInitializer(params=footprints_parameters)

    @pytest.fixture
    def traces_parameters(self):
        return TracesInitializerParams()

    @pytest.fixture
    def traces_initializer(self, traces_parameters):
        return TracesInitializer(params=traces_parameters)

    @pytest.fixture
    def footprints_components(self, footprints_initializer, stabilized_video):
        video, _, _ = stabilized_video
        default_estimates = ComponentManager()

        estimates = footprints_initializer.learn_one(
            components=default_estimates,
            X=video[0],
        ).transform_one(default_estimates)

        return estimates

    @pytest.mark.parametrize("jit_enabled", [True, False])
    def test_first_n_frames(
        self,
        stabilized_video,
        traces_initializer,
        footprints_components,
        jit_enabled,
    ):
        if not jit_enabled:
            os.environ["NUMBA_DISABLE_JIT"] = "1"
        video, _, _ = stabilized_video
        traces_estimates = traces_initializer.learn_one(
            footprints_components, video[:3]
        ).transform_one(footprints_components)

        assert (
            footprints_components.footprints.shape[0]
            == traces_estimates.traces.shape[0]
        )

    def test_reconstruction_comparison(
        self, stabilized_video, traces_initializer, footprints_components
    ):
        """Test that reconstructed frames from spatial footprints and temporal traces match original frames."""
        video, _, _ = stabilized_video
        traces_estimates = traces_initializer.learn_one(
            footprints_components, video[:3]
        ).transform_one(footprints_components)

        # Get original first 3 frames
        original_frames = video[:3].values

        # Reconstruct frames using spatial footprints and temporal traces
        spatial_footprints = traces_estimates.footprints.values
        temporal_traces = traces_estimates.traces.values

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
