from typing import Any

import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import Component
from cala.streaming.nodes.init.odl.component_stats import (
    ComponentStatsInitializer,
    ComponentStatsInitializerParams,
)
from cala.viz_util import Visualizer


class TestComponentStatsInitializer:
    """Test suite for ComponentStatsInitializer."""

    @pytest.fixture
    def sample_data(self) -> dict[str, Any]:
        """Create sample data for testing."""
        # Create sample dimensions
        n_components = 3
        height, width = 10, 10
        n_frames = 5

        # Create sample coordinates
        coords = {
            "id_": ("component", [f"id{i}" for i in range(n_components)]),
            "type_": (
                "component",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
        }

        # Create sample traces with known correlation pattern
        traces_data = np.array(
            [
                [1.0, 0.5, 0.0, -0.5, -1.0],  # Component 1
                [1.0, 0.5, 0.0, -0.5, -1.0],  # Component 2 (perfect correlate with 1)
                [-1.0, -0.5, 0.0, 0.5, 1.0],  # Component 3 (anti-correlated with 1&2)
            ]
        )
        traces = xr.DataArray(traces_data, dims=("component", "frames"), coords=coords)

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
    def initializer(self) -> ComponentStatsInitializer:
        """Create ComponentStatsInitializer instance."""
        return ComponentStatsInitializer(ComponentStatsInitializerParams())

    def test_initialization(self, initializer: ComponentStatsInitializer) -> None:
        """Test proper initialization."""
        assert isinstance(initializer.params, ComponentStatsInitializerParams)
        assert not hasattr(initializer, "component_stats_")  # Should not exist before learn_one

    def test_learn_one(
        self, initializer: ComponentStatsInitializer, sample_data: dict[str, Any]
    ) -> None:
        """Test learn_one method."""
        # Run learn_one
        initializer.learn_one(sample_data["traces"], sample_data["frames"])

        # Check that component_stats_ was created
        assert hasattr(initializer, "component_stats_")
        assert isinstance(initializer.component_stats_, xr.DataArray)

        # Check dimensions
        assert initializer.component_stats_.dims == ("component", "component'")
        assert initializer.component_stats_.shape == (
            sample_data["n_components"],
            sample_data["n_components"],
        )

        # Check coordinates
        assert "id_" in initializer.component_stats_.coords
        assert "type_" in initializer.component_stats_.coords
        assert initializer.component_stats_.coords["type_"].values.tolist() == [
            Component.NEURON.value,
            Component.NEURON.value,
            Component.BACKGROUND.value,
        ]

    def test_transform_one(
        self, initializer: ComponentStatsInitializer, sample_data: dict[str, Any]
    ) -> None:
        """Test transform_one method."""
        # First learn
        initializer.learn_one(sample_data["traces"], sample_data["frames"])

        # Then transform
        result = initializer.transform_one()

        # Check result type
        assert isinstance(result, xr.DataArray)

        # Check dimensions
        assert result.dims == ("component", "component'")
        assert result.shape == (
            sample_data["n_components"],
            sample_data["n_components"],
        )

    @pytest.mark.viz
    def test_computation_correctness(
        self,
        initializer: ComponentStatsInitializer,
        sample_data: dict[str, Any],
        visualizer: Visualizer,
    ) -> None:
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

        # Visualize correlations if enabled
        if visualizer is not None:
            visualizer.plot_trace_correlations(
                traces, name="trace_correlations", subdir="init/comp_stats"
            )
            visualizer.plot_component_stats(
                result, name="component_correlation_matrix", subdir="init/comp_stats"
            )

    def test_matrix_properties(
        self, initializer: ComponentStatsInitializer, sample_data: dict[str, Any]
    ) -> None:
        """Test mathematical properties of the correlation matrix."""
        # Run computation
        initializer.learn_one(sample_data["traces"], sample_data["frames"])
        result = initializer.transform_one()

        # Test symmetry
        assert np.allclose(result.values, result.values.T)

        # Test diagonal elements
        assert np.allclose(np.diag(result.values), 0.5)

    def test_coordinate_preservation(
        self, initializer: ComponentStatsInitializer, sample_data: dict[str, Any]
    ) -> None:
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
