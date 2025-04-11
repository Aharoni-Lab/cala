from typing import Any

import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import Component
from cala.streaming.nodes.init.odl.residuals import (
    ResidualInitializer,
    ResidualInitializerParams,
)
from cala.viz_util import Visualizer


class TestResidualInitializerParams:
    """Test suite for ResidualInitializerParams."""

    @pytest.fixture
    def default_params(self) -> ResidualInitializerParams:
        """Create default parameters instance."""
        return ResidualInitializerParams()

    def test_validation_valid_params(self) -> None:
        """Test validation with valid parameters."""
        params = ResidualInitializerParams(buffer_length=10)
        params.validate()  # Should not raise


class TestResidualInitializer:
    """Test suite for ResidualInitializer."""

    @pytest.fixture
    def sample_data(self) -> dict[str, Any]:
        """Create sample data for testing."""
        # Create sample dimensions
        n_components = 3
        height, width = 5, 5
        n_frames = 15  # Larger than default buffer to test buffering

        # Create sample coordinates
        coords = {
            "id_": ("component", [f"id{i}" for i in range(n_components)]),
            "type_": (
                "component",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
        }

        footprints = xr.DataArray(
            np.zeros((n_components, height, width)),
            dims=("component", "height", "width"),
            coords=coords,
        )
        footprints[0, 0:2, 0:2] = 1
        footprints[1, 1:4, 1:4] = 3
        footprints[2, 3:5, 3:5] = 2

        traces = xr.DataArray(
            np.zeros((n_components, n_frames)),
            dims=("component", "frame"),
            coords=coords,
        )
        traces[0, :] = [1 for _ in range(n_frames)]
        traces[1, :] = [i for i in range(n_frames)]
        traces[2, :] = [n_frames - i for i in range(n_frames)]

        residual = xr.DataArray(
            np.zeros((n_frames, height, width)), dims=("frame", "height", "width")
        )
        for i in range(n_frames):
            residual[i, :, i % width] = 3

        denoised_movie = footprints @ traces
        movie = denoised_movie + residual

        return {
            "movie": movie.transpose("frame", "height", "width"),
            "denoised": denoised_movie.transpose("frame", "height", "width"),
            "footprints": footprints,
            "traces": traces,
            "residual": residual.transpose("frame", "height", "width"),
        }

    @pytest.fixture
    def initializer(self) -> ResidualInitializer:
        """Create ResidualInitializer instance with small buffer for testing."""
        params = ResidualInitializerParams(buffer_length=5)
        return ResidualInitializer(params)

    def test_initialization(self, initializer: ResidualInitializer) -> None:
        """Test proper initialization."""
        assert isinstance(initializer.params, ResidualInitializerParams)
        assert not hasattr(initializer, "residual_")  # Should not exist before learn_one

    def test_learn_one(self, initializer: ResidualInitializer, sample_data: dict[str, Any]) -> None:
        """Test learn_one method."""
        # Run learn_one
        initializer.learn_one(
            sample_data["footprints"], sample_data["traces"], sample_data["movie"]
        )

        # Check that residual_ was created
        assert hasattr(initializer, "residual_")
        assert isinstance(initializer.residual_, xr.DataArray)

        # Check dimensions
        assert initializer.residual_.sizes == {
            "frame": min(initializer.params.buffer_length, sample_data["movie"].sizes["frame"]),
            "width": sample_data["movie"].sizes["width"],
            "height": sample_data["movie"].sizes["height"],
        }

    def test_transform_one(
        self, initializer: ResidualInitializer, sample_data: dict[str, Any]
    ) -> None:
        """Test transform_one method."""
        # First learn
        initializer.learn_one(
            sample_data["footprints"], sample_data["traces"], sample_data["movie"]
        )

        # Then transform
        result = initializer.transform_one()

        # Check result type
        assert isinstance(result, xr.DataArray)
        assert set(result.dims) == {"frame", "width", "height"}

    @pytest.mark.viz
    def test_computation_correctness(
        self, sample_data: dict[str, Any], visualizer: Visualizer
    ) -> None:
        """Test the correctness of the residual computation."""
        # Prepare data
        sample_movie = sample_data["movie"]
        sample_denoised = sample_data["denoised"]
        sample_footprints = sample_data["footprints"]
        sample_traces = sample_data["traces"]
        sample_residual = sample_data["residual"]

        isitclean = sample_movie - sample_denoised - sample_residual

        visualizer.plot_footprints(sample_footprints, subdir="init/resid")
        visualizer.plot_traces(sample_traces, subdir="init/resid")
        visualizer.save_video_frames(
            [
                (sample_movie, "movie"),
                (sample_denoised, "denoised"),
                (sample_residual, "residual"),
                (isitclean, "isitclean"),
            ],
            subdir="init/resid",
        )

        initializer = ResidualInitializer(
            ResidualInitializerParams(buffer_length=len(sample_movie))
        )

        # Run computation
        initializer.learn_one(sample_footprints, sample_traces, sample_movie)
        result = initializer.transform_one()

        assert np.array_equal(
            sample_residual.transpose("frame", "height", "width"),
            result.transpose("frame", "height", "width"),
        )

    def test_buffer_length(self, sample_data: dict[str, Any]) -> None:
        """Test that buffer length is properly enforced."""
        # Create initializer with small buffer
        small_buffer_initializer = ResidualInitializer(ResidualInitializerParams(buffer_length=3))

        # Run computation
        small_buffer_initializer.learn_one(
            sample_data["footprints"], sample_data["traces"], sample_data["movie"]
        )
        result = small_buffer_initializer.transform_one()

        # Check buffer length
        assert result.sizes["frame"] == 3

    def test_invalid_input_handling(self, initializer: ResidualInitializer) -> None:
        """Test handling of invalid inputs."""
        # Test with mismatched dimensions
        invalid_footprints = xr.DataArray(
            np.random.rand(3, 8, 8),  # Different spatial dimensions
            dims=("component", "height", "width"),
            coords={
                "id_": ("component", ["id0", "id1", "id2"]),
                "type_": (
                    "component",
                    [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
                ),
            },
        )
        invalid_traces = xr.DataArray(
            np.random.rand(3, 10),
            dims=("component", "frame"),
            coords={
                "id_": ("component", ["id0", "id1", "id2"]),
                "type_": (
                    "component",
                    [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
                ),
            },
        )
        invalid_frames = xr.DataArray(
            np.random.rand(5, 10, 10),  # Mismatched with traces
            dims=("frame", "height", "width"),
        )

        with pytest.raises(ValueError):  # Should raise some kind of error
            initializer.learn_one(invalid_footprints, invalid_traces, invalid_frames)
