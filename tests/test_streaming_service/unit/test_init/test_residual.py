import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import Traces, Footprints
from cala.streaming.init.odl.residual_buffer import (
    ResidualInitializer,
    ResidualInitializerParams,
)
from cala.streaming.stores.odl import Residual


class TestResidualInitializerParams:
    """Test suite for ResidualInitializerParams."""

    @pytest.fixture
    def default_params(self):
        """Create default parameters instance."""
        return ResidualInitializerParams()

    def test_default_values(self, default_params):
        """Test default parameter values."""
        assert default_params.component_axis == "components"
        assert default_params.id_coordinates == "id_"
        assert default_params.type_coordinates == "type_"
        assert default_params.frames_axis == "frame"
        assert default_params.spatial_axes == ("height", "width")
        assert default_params.buffer_length == 50

    def test_validation_valid_params(self):
        """Test validation with valid parameters."""
        params = ResidualInitializerParams(spatial_axes=("y", "x"), buffer_length=10)
        params.validate()  # Should not raise

    def test_validation_invalid_params(self):
        """Test validation with invalid parameters."""
        # Test with wrong spatial axes type
        with pytest.raises(ValueError):
            params = ResidualInitializerParams(spatial_axes=["height", "width"])
            params.validate()

        # Test with wrong spatial axes length
        with pytest.raises(ValueError):
            params = ResidualInitializerParams(spatial_axes=("height",))
            params.validate()

        # Test with invalid buffer length
        with pytest.raises(ValueError):
            params = ResidualInitializerParams(buffer_length=0)
            params.validate()

        with pytest.raises(ValueError):
            params = ResidualInitializerParams(buffer_length=-1)
            params.validate()


class TestResidualInitializer:
    """Test suite for ResidualInitializer."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        # Create sample dimensions
        n_components = 3
        height, width = 10, 10
        n_frames = 15  # Larger than default buffer to test buffering

        # Create sample coordinates
        coords = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": ("components", ["neuron", "neuron", "background"]),
        }

        # Create sample footprints
        footprints_data = np.random.rand(n_components, height, width)
        footprints = Footprints(
            footprints_data, dims=("components", "height", "width"), coords=coords
        )

        # Create sample traces
        traces_data = np.random.rand(n_components, n_frames)
        traces = Traces(traces_data, dims=("components", "frames"), coords=coords)

        # Create sample frames
        frames_data = np.random.rand(n_frames, height, width)
        frames = xr.DataArray(frames_data, dims=("frame", "height", "width"))

        return {
            "footprints": footprints,
            "traces": traces,
            "frames": frames,
            "n_components": n_components,
            "height": height,
            "width": width,
            "n_frames": n_frames,
        }

    @pytest.fixture
    def initializer(self):
        """Create ResidualInitializer instance with small buffer for testing."""
        params = ResidualInitializerParams(buffer_length=5)
        return ResidualInitializer(params)

    def test_initialization(self, initializer):
        """Test proper initialization."""
        assert isinstance(initializer.params, ResidualInitializerParams)
        assert not hasattr(
            initializer, "residual_"
        )  # Should not exist before learn_one

    def test_learn_one(self, initializer, sample_data):
        """Test learn_one method."""
        # Run learn_one
        initializer.learn_one(
            sample_data["footprints"], sample_data["traces"], sample_data["frames"]
        )

        # Check that residual_ was created
        assert hasattr(initializer, "residual_")
        assert isinstance(initializer.residual_, xr.DataArray)

        # Check dimensions
        assert initializer.residual_.dims == ("height", "width", "frame")
        assert initializer.residual_.shape == (
            sample_data["height"],
            sample_data["width"],
            min(initializer.params.buffer_length, sample_data["n_frames"]),
        )

    def test_transform_one(self, initializer, sample_data):
        """Test transform_one method."""
        # First learn
        initializer.learn_one(
            sample_data["footprints"], sample_data["traces"], sample_data["frames"]
        )

        # Then transform
        result = initializer.transform_one()

        # Check result type
        assert isinstance(result, Residual)
        assert result.dims == ("height", "width", "frame")

    def test_computation_correctness(self, initializer, sample_data):
        """Test the correctness of the residual computation."""
        # Prepare data
        footprints = sample_data["footprints"]
        traces = sample_data["traces"]
        frames = sample_data["frames"]

        # Run computation
        initializer.learn_one(footprints, traces, frames)
        result = initializer.transform_one()

        # Manual computation for verification
        Y = frames.values.reshape(-1, frames.shape[0])
        A = footprints.values.reshape(footprints.shape[0], -1).T
        C = traces.values

        expected_R = Y - A @ C
        start_idx = max(0, frames.shape[0] - initializer.params.buffer_length)
        expected_R = expected_R[:, start_idx:]
        expected_R = expected_R.reshape(frames.shape[1], frames.shape[2], -1)

        # Compare results
        assert np.allclose(result.values, expected_R)

    def test_buffer_length(self, sample_data):
        """Test that buffer length is properly enforced."""
        # Create initializer with small buffer
        small_buffer_initializer = ResidualInitializer(
            ResidualInitializerParams(buffer_length=3)
        )

        # Run computation
        small_buffer_initializer.learn_one(
            sample_data["footprints"], sample_data["traces"], sample_data["frames"]
        )
        result = small_buffer_initializer.transform_one()

        # Check buffer length
        assert result.sizes["frame"] == 3

    def test_invalid_input_handling(self, initializer):
        """Test handling of invalid inputs."""
        # Test with mismatched dimensions
        invalid_footprints = Footprints(
            np.random.rand(3, 8, 8),  # Different spatial dimensions
            dims=("components", "height", "width"),
            coords={
                "id_": ("components", ["id0", "id1", "id2"]),
                "type_": ("components", ["neuron", "neuron", "background"]),
            },
        )
        invalid_traces = Traces(
            np.random.rand(3, 10),
            dims=("components", "frames"),
            coords={
                "id_": ("components", ["id0", "id1", "id2"]),
                "type_": ("components", ["neuron", "neuron", "background"]),
            },
        )
        invalid_frames = xr.DataArray(
            np.random.rand(5, 10, 10),  # Mismatched with traces
            dims=("frame", "height", "width"),
        )

        with pytest.raises(Exception):  # Should raise some kind of error
            initializer.learn_one(invalid_footprints, invalid_traces, invalid_frames)
