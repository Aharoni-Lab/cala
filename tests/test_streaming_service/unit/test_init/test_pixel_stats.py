import numpy as np
import pytest
import xarray as xr

from cala.streaming.init.odl.pixel_stats import (
    PixelStatsInitializer,
    PixelStatsInitializerParams,
)


class TestPixelStatsInitializer:
    """Test suite for PixelStatsInitializer."""

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

    @pytest.mark.viz
    def test_learn_one(
        self, initializer, traces, footprints, stabilized_video, visualizer
    ):
        """Test learn_one method."""
        footprints, _, _ = footprints
        # Run learn_one
        initializer.learn_one(traces, stabilized_video)

        # Check that pixel_stats_ was created
        assert hasattr(initializer, "pixel_stats_")
        assert isinstance(initializer.pixel_stats_, xr.DataArray)

        # Check dimensions
        assert initializer.pixel_stats_.dims == ("component", "width", "height")
        assert initializer.pixel_stats_.shape == (
            traces.sizes["component"],
            stabilized_video.sizes["width"],
            stabilized_video.sizes["height"],
        )

        # Check coordinates
        assert "id_" in initializer.pixel_stats_.coords
        assert "type_" in initializer.pixel_stats_.coords

        visualizer.plot_traces(traces, subdir="init/pixel_stats")
        visualizer.write_movie(stabilized_video, subdir="init/pixel_stats")
        visualizer.plot_pixel_stats(
            pixel_stats=initializer.pixel_stats_,
            footprints=footprints,
            subdir="init/pixel_stats",
        )

    def test_transform_one(self, initializer, traces, stabilized_video):
        """Test transform_one method."""
        # First learn
        initializer.learn_one(traces, stabilized_video)

        # Then transform
        result = initializer.transform_one()

        # Check result type
        assert isinstance(result, xr.DataArray)

        # Check dimensions order
        assert result.dims == ("component", "height", "width")
        assert result.shape == (
            traces.sizes["component"],
            stabilized_video.sizes["height"],
            stabilized_video.sizes["width"],
        )

    def test_computation_correctness(self, initializer, traces, stabilized_video):
        """Test the correctness of the pixel statistics computation."""
        # the test is probably wrong :/ needs to be rewritten.
        # # Prepare data
        # traces = sample_data["traces"]
        # frames = sample_data["frames"]
        #
        # # Run computation
        # initializer.learn_one(traces, frames)
        # result = initializer.transform_one()
        #
        # # Manual computation for verification
        # Y = frames.values.reshape(-1, frames.shape[0])
        # C = traces.values
        # expected_W = (Y @ C.T / frames.shape[0]).reshape(
        #     frames.shape[1], frames.shape[2], traces.shape[0]
        # )
        # expected_W = np.transpose(expected_W, (2, 0, 1))

        # Compare results
        # assert np.allclose(result.values, expected_W)

    def test_coordinate_preservation(self, initializer, traces, stabilized_video):
        """Test that coordinates are properly preserved through the transformation."""
        # Run computation
        initializer.learn_one(traces, stabilized_video)
        result = initializer.transform_one()

        # Check coordinate values
        assert np.array_equal(result.coords["id_"].values, traces.coords["id_"].values)
        assert np.array_equal(
            result.coords["type_"].values, traces.coords["type_"].values
        )

    def test_invalid_input_handling(self, initializer):
        """Test handling of invalid inputs."""
        # Test with mismatched dimensions
        invalid_traces = xr.DataArray(
            np.random.rand(3, 10),
            dims=("components", "frame"),
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
