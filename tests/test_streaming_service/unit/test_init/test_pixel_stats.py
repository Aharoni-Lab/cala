import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import Component
from cala.streaming.nodes.init.odl.pixel_stats import (
    PixelStatsInitializer,
    PixelStatsInitializerParams,
)
from cala.viz_util import Visualizer


class TestPixelStatsInitializer:
    """Test suite for PixelStatsInitializer."""

    @pytest.fixture
    def initializer(self) -> PixelStatsInitializer:
        """Create PixelStatsInitializer instance."""
        return PixelStatsInitializer(PixelStatsInitializerParams())

    def test_initialization(self, initializer: PixelStatsInitializer) -> None:
        """Test proper initialization."""
        assert isinstance(initializer.params, PixelStatsInitializerParams)
        assert not hasattr(initializer, "pixel_stats_")  # Should not exist before learn_one

    @pytest.mark.viz
    def test_learn_one(
        self,
        initializer: PixelStatsInitializer,
        mini_traces: xr.DataArray,
        mini_footprints: xr.DataArray,
        mini_denoised: xr.DataArray,
        visualizer: Visualizer,
    ) -> None:
        """Test learn_one method."""
        # Run learn_one
        initializer.learn_one(mini_traces, mini_denoised)

        # Check that pixel_stats_ was created
        assert hasattr(initializer, "pixel_stats_")
        assert isinstance(initializer.pixel_stats_, xr.DataArray)

        # Check dimensions
        assert initializer.pixel_stats_.sizes == {
            "component": mini_traces.sizes["component"],
            "width": mini_denoised.sizes["width"],
            "height": mini_denoised.sizes["height"],
        }

        # Check coordinates
        assert "id_" in initializer.pixel_stats_.coords
        assert "type_" in initializer.pixel_stats_.coords

        visualizer.plot_traces(mini_traces, subdir="init/pixel_stats")
        visualizer.write_movie(mini_denoised, subdir="init/pixel_stats")
        visualizer.plot_pixel_stats(
            pixel_stats=initializer.pixel_stats_.transpose(*mini_footprints.dims),
            footprints=mini_footprints,
            subdir="init/pixel_stats",
        )

    def test_transform_one(
        self,
        initializer: PixelStatsInitializer,
        traces: xr.DataArray,
        stabilized_video: xr.DataArray,
    ) -> None:
        """Test transform_one method."""
        # First learn
        initializer.learn_one(traces, stabilized_video)

        # Then transform
        result = initializer.transform_one()

        # Check result type
        assert isinstance(result, xr.DataArray)

        # Check dimensions order
        assert result.sizes == {
            "component": traces.sizes["component"],
            "width": stabilized_video.sizes["width"],
            "height": stabilized_video.sizes["height"],
        }

    @pytest.mark.viz
    def test_sanity_check(self, initializer: PixelStatsInitializer, visualizer: Visualizer) -> None:
        """Test the correctness of the pixel statistics computation."""
        video = xr.DataArray(np.zeros((2, 2, 3)), dims=("height", "width", "frame"))
        video[0, 0, :] = [1, 2, 3]
        video[1, 1, :] = [3, 2, 1]
        video[0, 1, :] = [1, 2, 1]

        traces = xr.DataArray(
            np.zeros((2, 3)),
            dims=("component", "frame"),
            coords={
                "id_": ("component", ["comp1", "comp2"]),
                "type_": ("component", [Component.NEURON.value, Component.NEURON.value]),
            },
        )
        traces[0, :] = [1, 2, 3]
        traces[1, :] = [3, 2, 1]

        # Run computation
        initializer.learn_one(traces, video)
        result = initializer.transform_one().transpose("component", "width", "height")

        label = (video @ traces).transpose("component", "width", "height") / video.sizes["frame"]

        visualizer.plot_traces(traces, subdir="init/pixel_stats/sanity_check")
        visualizer.plot_pixel_stats(result, subdir="init/pixel_stats/sanity_check")

        assert np.array_equal(result, label)

    @pytest.mark.viz
    def test_sanity_check_2(
        self,
        mini_denoised: xr.DataArray,
        mini_traces: xr.DataArray,
        mini_footprints: xr.DataArray,
        initializer: PixelStatsInitializer,
        visualizer: Visualizer,
    ) -> None:
        """Test the correctness of the pixel statistics computation."""

        # Run computation
        initializer.learn_one(mini_traces, mini_denoised)
        result = initializer.transform_one().transpose("component", "width", "height")

        label = (mini_denoised @ mini_traces).transpose(
            "component", "width", "height"
        ) / mini_denoised.sizes["frame"]

        visualizer.plot_footprints(mini_footprints, subdir="init/pixel_stats/sanity_check_2")
        visualizer.plot_traces(mini_traces, subdir="init/pixel_stats/sanity_check_2")
        visualizer.plot_pixel_stats(
            result,
            mini_footprints,
            subdir="init/pixel_stats/sanity_check_2",
            name="result",
        )
        visualizer.plot_pixel_stats(
            label,
            mini_footprints,
            subdir="init/pixel_stats/sanity_check_2",
            name="label",
        )

        assert np.allclose(result, label, atol=1e-3)

    def test_coordinate_preservation(
        self,
        initializer: PixelStatsInitializer,
        traces: xr.DataArray,
        stabilized_video: xr.DataArray,
    ) -> None:
        """Test that coordinates are properly preserved through the transformation."""
        # Run computation
        initializer.learn_one(traces, stabilized_video)
        result = initializer.transform_one()

        # Check coordinate values
        assert np.array_equal(result.coords["id_"].values, traces.coords["id_"].values)
        assert np.array_equal(result.coords["type_"].values, traces.coords["type_"].values)

    def test_invalid_input_handling(self, initializer: PixelStatsInitializer) -> None:
        """Test handling of invalid inputs."""
        # Test with mismatched dimensions
        invalid_traces = xr.DataArray(
            np.random.rand(3, 10),
            dims=("components", "frame"),
            coords={
                "id_": ("components", ["id0", "id1", "id2"]),
                "type_": (
                    "components",
                    [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
                ),
            },
        )
        invalid_frames = xr.DataArray(
            np.random.rand(5, 8, 8),  # Different spatial dimensions
            dims=("frame", "height", "width"),
        )

        with pytest.raises(ValueError):  # Should raise some kind of error
            initializer.learn_one(invalid_traces, invalid_frames)
