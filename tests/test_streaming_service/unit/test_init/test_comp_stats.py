import numpy as np
import pytest
import xarray as xr

from cala.streaming.init.odl.component_stats import (
    ComponentStatsInitializer,
    ComponentStatsInitializerParams,
)


class TestComponentStatsInitializerParams:
    """Test suite for ComponentStatsInitializerParams."""

    @pytest.fixture
    def default_params(self):
        """Create default parameters instance."""
        return ComponentStatsInitializerParams()

    def test_default_values(self, default_params):
        """Test default parameter values."""
        assert default_params.component_axis == "components"
        assert default_params.id_coordinates == "id_"
        assert default_params.type_coordinates == "type_"
        assert default_params.frames_axis == "frame"
        assert default_params.spatial_axes == ("height", "width")

    def test_validation_valid_spatial_axes(self):
        """Test validation with valid spatial axes."""
        params = ComponentStatsInitializerParams(spatial_axes=("y", "x"))
        params.validate()  # Should not raise

    def test_validation_invalid_spatial_axes(self):
        """Test validation with invalid spatial axes."""
        # Test with wrong type
        with pytest.raises(ValueError):
            params = ComponentStatsInitializerParams(spatial_axes=["height", "width"])
            params.validate()

        # Test with wrong length
        with pytest.raises(ValueError):
            params = ComponentStatsInitializerParams(spatial_axes=("height",))
            params.validate()


class TestComponentStatsInitializer:
    """Test suite for ComponentStatsInitializer."""

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

        # Create sample traces with known correlation pattern
        traces_data = np.array(
            [
                [1.0, 0.5, 0.0, -0.5, -1.0],  # Component 1
                [
                    1.0,
                    0.5,
                    0.0,
                    -0.5,
                    -1.0,
                ],  # Component 2 (perfectly correlated with 1)
                [-1.0, -0.5, 0.0, 0.5, 1.0],  # Component 3 (anti-correlated with 1&2)
            ]
        )
        traces = xr.DataArray(traces_data, dims=("components", "frames"), coords=coords)

        # Create sample frames
        frames_data = np.random.rand(n_frames, height, width)
        frames = xr.DataArray(frames_data, dims=("frame", "height", "width"))

        return {
            "traces": traces,
            "frames": frames,
            "n_components": n_components,
            "n_frames": n_frames,
        }

    @pytest.fixture
    def initializer(self):
        """Create ComponentStatsInitializer instance."""
        return ComponentStatsInitializer(ComponentStatsInitializerParams())

    def test_initialization(self, initializer):
        """Test proper initialization."""
        assert isinstance(initializer.params, ComponentStatsInitializerParams)
        assert not hasattr(
            initializer, "component_stats_"
        )  # Should not exist before learn_one

    def test_learn_one(self, initializer, sample_data):
        """Test learn_one method."""
        # Run learn_one
        initializer.learn_one(sample_data["traces"], sample_data["frames"])

        # Check that component_stats_ was created
        assert hasattr(initializer, "component_stats_")
        assert isinstance(initializer.component_stats_, xr.DataArray)

        # Check dimensions
        assert initializer.component_stats_.dims == ("components", "components'")
        assert initializer.component_stats_.shape == (
            sample_data["n_components"],
            sample_data["n_components"],
        )

        # Check coordinates
        assert "id_" in initializer.component_stats_.coords
        assert "type_" in initializer.component_stats_.coords
        assert initializer.component_stats_.coords["type_"].values.tolist() == [
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
        assert isinstance(result, xr.DataArray)

        # Check dimensions
        assert result.dims == ("components", "components'")
        assert result.shape == (
            sample_data["n_components"],
            sample_data["n_components"],
        )

    def test_computation_correctness(self, initializer, sample_data):
        """Test the correctness of the component correlation computation."""
        # Prepare data
        traces = sample_data["traces"]
        frames = sample_data["frames"]

        # Run computation
        initializer.learn_one(traces, frames)
        result = initializer.transform_one()

        # Manual computation for verification
        C = traces.values
        expected_M = C @ C.T / frames.shape[0]

        # Compare results
        assert np.allclose(result.values, expected_M)

        # Check specific correlation patterns from our constructed data
        assert np.allclose(result.values[0, 1], 0.5)  # Perfect correlation
        assert np.allclose(result.values[0, 2], -0.5)  # Perfect anti-correlation
        assert np.allclose(np.diag(result.values), 0.5)  # Self-correlation

    def test_matrix_properties(self, initializer, sample_data):
        """Test mathematical properties of the correlation matrix."""
        # Run computation
        initializer.learn_one(sample_data["traces"], sample_data["frames"])
        result = initializer.transform_one()

        # Test symmetry
        assert np.allclose(result.values, result.values.T)

        # Test diagonal elements
        assert np.allclose(np.diag(result.values), 0.5)

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
