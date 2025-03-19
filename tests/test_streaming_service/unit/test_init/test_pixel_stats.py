import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import Traces
from cala.streaming.init.odl.pixel_stats import (
    PixelStatsInitializer,
    PixelStatsInitializerParams,
)
from cala.streaming.stores.odl import PixelStats


class TestPixelStatsInitializerParams:
    """Test suite for PixelStatsInitializerParams."""

    @pytest.fixture
    def default_params(self):
        """Create default parameters instance."""
        return PixelStatsInitializerParams()

    def test_default_values(self, default_params):
        """Test default parameter values."""
        assert default_params.component_axis == "components"
        assert default_params.id_coordinates == "id_"
        assert default_params.type_coordinates == "type_"
        assert default_params.frames_axis == "frame"
        assert default_params.spatial_axes == ("height", "width")

    def test_validation_valid_spatial_axes(self):
        """Test validation with valid spatial axes."""
        params = PixelStatsInitializerParams(spatial_axes=("y", "x"))
        params.validate()  # Should not raise

    def test_validation_invalid_spatial_axes(self):
        """Test validation with invalid spatial axes."""
        # Test with wrong type
        with pytest.raises(ValueError):
            params = PixelStatsInitializerParams(spatial_axes=["height", "width"])
            params.validate()

        # Test with wrong length
        with pytest.raises(ValueError):
            params = PixelStatsInitializerParams(spatial_axes=("height",))
            params.validate()


class TestPixelStatsInitializer:
    """Test suite for PixelStatsInitializer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample dimensions
        n_components = 3
        height, width = 10, 10
        n_frames = 5

        # Create sample coordinates
        coords = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": ("components", ["neuron", "neuron", "background"]),
        }

        # Create sample traces
        traces_data = np.random.rand(n_components, n_frames)
        traces = Traces(traces_data, dims=("components", "frames"), coords=coords)

        # Create sample frames
        frames_data = np.random.rand(n_frames, height, width)
        frames = xr.DataArray(frames_data, dims=("frame", "height", "width"))

        return {
            "traces": traces,
            "frames": frames,
            "n_components": n_components,
            "height": height,
            "width": width,
            "n_frames": n_frames,
        }

    @pytest.fixture
    def initializer(self):
        """Create PixelStatsInitializer instance."""
        return PixelStatsInitializer(PixelStatsInitializerParams())

    def test_initialization(self, initializer):
        """Test proper initialization."""
        assert isinstance(initializer.params, PixelStatsInitializerParams)
        assert not hasattr(
            initializer, "pixel_stats_"
        )  # Should not exist before learn_one

    def test_learn_one(self, initializer, sample_data):
        """Test learn_one method."""
        # Run learn_one
        initializer.learn_one(sample_data["traces"], sample_data["frames"])

        # Check that pixel_stats_ was created
        assert hasattr(initializer, "pixel_stats_")
        assert isinstance(initializer.pixel_stats_, xr.DataArray)

        # Check dimensions
        assert initializer.pixel_stats_.dims == ("height", "width", "components")
        assert initializer.pixel_stats_.shape == (
            sample_data["height"],
            sample_data["width"],
            sample_data["n_components"],
        )

        # Check coordinates
        assert "id_" in initializer.pixel_stats_.coords
        assert "type_" in initializer.pixel_stats_.coords
        assert list(initializer.pixel_stats_.coords["type_"].values) == [
            "neuron",
            "neuron",
            "background",
        ]

    def test_transform_one(self, initializer, sample_data):
        """Test transform_one method."""
        # First learn
        initializer.learn_one(sample_data["traces"], sample_data["frames"])

        # Then transform
        result = initializer.transform_one()

        # Check result type
        assert isinstance(result, PixelStats)

        # Check dimensions order
        assert result.dims == ("components", "height", "width")
        assert result.shape == (
            sample_data["n_components"],
            sample_data["height"],
            sample_data["width"],
        )

    def test_computation_correctness(self, initializer, sample_data):
        """Test the correctness of the pixel statistics computation."""
        # Prepare data
        traces = sample_data["traces"]
        frames = sample_data["frames"]

        # Run computation
        initializer.learn_one(traces, frames)
        result = initializer.transform_one()

        # Manual computation for verification
        Y = frames.values.reshape(-1, frames.shape[0])
        C = traces.values
        expected_W = (Y @ C.T / frames.shape[0]).reshape(
            frames.shape[1], frames.shape[2], traces.shape[0]
        )
        expected_W = np.transpose(expected_W, (2, 0, 1))

        # Compare results
        assert np.allclose(result.values, expected_W)

    def test_coordinate_preservation(self, initializer, sample_data):
        """Test that coordinates are properly preserved through the transformation."""
        # Run computation
        initializer.learn_one(sample_data["traces"], sample_data["frames"])
        result = initializer.transform_one()

        # Check coordinate values
        assert np.array_equal(
            result.coords["id_"].values, sample_data["traces"].coords["id_"].values
        )
        assert np.array_equal(
            result.coords["type_"].values, sample_data["traces"].coords["type_"].values
        )

    def test_invalid_input_handling(self, initializer):
        """Test handling of invalid inputs."""
        # Test with mismatched dimensions
        invalid_traces = Traces(
            np.random.rand(3, 10),
            dims=("components", "frames"),
            coords={
                "id_": ("components", ["id0", "id1", "id2"]),
                "type_": ("components", ["neuron", "neuron", "background"]),
            },
        )
        invalid_frames = xr.DataArray(
            np.random.rand(5, 8, 8),  # Different spatial dimensions
            dims=("frame", "height", "width"),
        )

        with pytest.raises(Exception):  # Should raise some kind of error
            initializer.learn_one(invalid_traces, invalid_frames)
