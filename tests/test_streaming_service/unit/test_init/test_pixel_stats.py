from dataclasses import dataclass

import numpy as np
import pytest
import xarray as xr

from cala.streaming.init.odl.pixel_stats import (
    PixelStatsInitializer,
    PixelStatsInitializerParams,
)


@dataclass
class TestParams:
    n_components = 5
    height = 10
    width = 10
    n_frames = 5


class TestPixelStatsInitializer:
    """Test suite for PixelStatsInitializer."""

    @pytest.fixture
    def p(self):
        return TestParams()

    @pytest.fixture
    def coords(self, p):
        return {
            "id_": ("component", [f"id{i}" for i in range(p.n_components)]),
            "type_": ("component", ["neuron"] * (p.n_components - 1) + ["background"]),
        }

    @pytest.fixture
    def sample_footprints(self, p, coords):
        """Create sample footprints for testing.

        Creates a set of footprints with known overlap patterns:
        - Components 0 and 1 overlap
        - Component 2 is isolated
        - Components 3 and 4 overlap
        """
        # Create empty footprints
        footprints_data = np.zeros((p.n_components, p.height, p.width))

        # Set up specific overlap patterns
        footprints_data[0, 0:5, 0:5] = 1  # Component 0
        footprints_data[1, 3:8, 3:8] = 1  # Component 1 (overlaps with 0)
        footprints_data[2, 8:10, 8:10] = 1  # Component 2 (isolated)
        footprints_data[3, 0:3, 8:10] = 1  # Component 3
        footprints_data[4, 1:4, 7:9] = 1  # Component 4 (overlaps with 3)

        return xr.DataArray(
            footprints_data, dims=("component", "height", "width"), coords=coords
        )

    @pytest.fixture
    def sample_traces(self, p, coords):
        traces = xr.DataArray(
            np.zeros((p.n_components, p.n_frames)),
            dims=("component", "frame"),
            coords=coords,
        )
        traces[0, :] = [1 for _ in range(p.n_frames)]
        traces[1, :] = [i for i in range(p.n_frames)]
        traces[2, :] = [p.n_frames - 1 - i for i in range(p.n_frames)]
        traces[3, :] = [abs((p.n_frames - 1) / 2 - i) for i in range(p.n_frames)]
        traces[4, :] = np.random.rand(p.n_frames)

        return traces

    @pytest.fixture
    def sample_residuals(self, p):
        residual = xr.DataArray(
            np.zeros((p.n_frames, p.height, p.width)), dims=("frame", "height", "width")
        )
        for i in range(p.n_frames):
            residual[i, :, i % p.width] = 3

        return residual

    @pytest.fixture
    def sample_denoised(self, sample_footprints, sample_traces):
        return (sample_footprints @ sample_traces).transpose("frame", "height", "width")

    @pytest.fixture
    def sample_movie(self, sample_denoised, sample_residuals):
        return (sample_denoised + sample_residuals).transpose(
            "frame", "height", "width"
        )

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

    @pytest.mark.viz
    def test_sanity_check(self, initializer, visualizer):
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
                "type_": ("component", ["neuron", "neuron"]),
            },
        )
        traces[0, :] = [1, 2, 3]
        traces[1, :] = [3, 2, 1]

        # Run computation
        initializer.learn_one(traces, video)
        result = initializer.transform_one().transpose("component", "width", "height")

        label = (video @ traces).transpose(
            "component", "width", "height"
        ) / video.sizes["frame"]

        visualizer.plot_traces(traces, subdir="init/pixel_stats/sanity_check")
        visualizer.plot_pixel_stats(result, subdir="init/pixel_stats/sanity_check")

        assert np.array_equal(result, label)

    @pytest.mark.viz
    def test_sanity_check_2(
        self, sample_denoised, sample_traces, sample_footprints, initializer, visualizer
    ):
        """Test the correctness of the pixel statistics computation."""

        # Run computation
        initializer.learn_one(sample_traces, sample_denoised)
        result = initializer.transform_one().transpose("component", "width", "height")

        label = (sample_denoised @ sample_traces).transpose(
            "component", "width", "height"
        ) / sample_denoised.sizes["frame"]

        visualizer.plot_footprints(
            sample_footprints, subdir="init/pixel_stats/sanity_check_2"
        )
        visualizer.plot_traces(sample_traces, subdir="init/pixel_stats/sanity_check_2")
        visualizer.plot_pixel_stats(
            result, subdir="init/pixel_stats/sanity_check_2", name="result"
        )
        visualizer.plot_pixel_stats(
            label, subdir="init/pixel_stats/sanity_check_2", name="label"
        )

        assert np.allclose(result, label, atol=1e-3)

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
