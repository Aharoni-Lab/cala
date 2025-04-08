import dataclasses

import numpy as np
import pytest
import xarray as xr

from cala.streaming.nodes.preprocess import (
    Downsampler,
    DownsamplerParams,
)


class TestStreamingDownsampler:
    @pytest.fixture
    def default_params(self) -> DownsamplerParams:
        """Create default parameters for testing"""
        params = DownsamplerParams(
            method="mean",
            dimensions=["width", "height"],
            strides=[2, 3],
        )
        return params

    @pytest.fixture
    def downsampler_mean(self, default_params: DownsamplerParams) -> Downsampler:
        """Create StreamingDownsampler instance with Gaussian method"""
        return Downsampler(default_params)

    @pytest.fixture
    def downsampler_subset(self, default_params: DownsamplerParams) -> Downsampler:
        """Create StreamingDownsampler instance with median method"""
        params = dataclasses.replace(default_params, method="subset")
        return Downsampler(params)

    def test_initialization(self, default_params: DownsamplerParams) -> None:
        """Test proper initialization of StreamingDownsampler"""
        downsampler = Downsampler(default_params)
        assert downsampler.params.method == default_params.method
        assert downsampler.params.dimensions == default_params.dimensions
        assert downsampler.params.strides == default_params.strides

    def test_parameter_validation(self) -> None:
        """Test parameter validation"""
        # Test invalid method
        with pytest.raises(ValueError, match="Downsampling method .* not understood"):
            Downsampler(DownsamplerParams(method="invalid"))
        with pytest.raises(ValueError, match="Length of 'dims' and 'strides' must be equal."):
            Downsampler(DownsamplerParams(strides=[1], dimensions=["a", "b"]))
        with pytest.raises(ValueError, match="'strides' must be greater than 1."):
            Downsampler(DownsamplerParams(strides=[-1, 0]))

    def test_mean_downsampling(
        self, downsampler_mean: Downsampler, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test downsampling using mean method"""
        video = raw_calcium_video
        frame = video[0]

        # Process frame
        result = downsampler_mean.transform_one(frame)

        # Check basic properties
        check_dimensions(frame, result, downsampler_mean.params)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

        # Verify downsampling
        coarsen_dims = {
            dim: stride
            for dim, stride in zip(
                downsampler_mean.params.dimensions,
                downsampler_mean.params.strides,
            )
        }
        expected = frame.coarsen(coarsen_dims, boundary="trim").mean(
            **downsampler_mean.params.kwargs
        )

        np.testing.assert_array_almost_equal(result, expected)

    def test_subset_downsampling(
        self, downsampler_subset: Downsampler, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test downsampling using mean method"""
        video = raw_calcium_video
        frame = video[0]

        # Process frame
        result = downsampler_subset.transform_one(frame)

        # Check basic properties
        check_dimensions(frame, result, downsampler_subset.params)
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

        # Verify downsampling
        indexers = {
            dim: slice(None, None, stride)
            for dim, stride in zip(
                downsampler_subset.params.dimensions,
                downsampler_subset.params.strides,
            )
        }
        expected = frame.isel(indexers)

        np.testing.assert_array_almost_equal(result, expected)

    def test_streaming_consistency(
        self, downsampler_mean: Downsampler, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test consistency of streaming downsampling"""
        video = raw_calcium_video
        frames = video[:5]

        # Process frames sequentially
        streaming_results = []
        for frame in frames:
            downsampler_mean.learn_one(frame)  # Should be a no-op
            streaming_results.append(downsampler_mean.transform_one(frame))

        # Process frames in batch
        coarsen_dims = {
            dim: stride
            for dim, stride in zip(
                downsampler_mean.params.dimensions,
                downsampler_mean.params.strides,
            )
        }
        batch_results = frames.coarsen(coarsen_dims, boundary="trim").mean(
            **downsampler_mean.params.kwargs
        )

        # Compare results
        for streaming, batch in zip(streaming_results, batch_results):
            np.testing.assert_array_almost_equal(streaming, batch)


def check_dimensions(
    frame: xr.DataArray, result: xr.DataArray, default_params: DownsamplerParams
) -> None:
    stride_assignment = {
        dim: strides
        for dim, strides in zip(
            default_params.dimensions,
            default_params.strides,
        )
    }

    expected = {name: frame[name].shape[0] / stride for name, stride in stride_assignment.items()}

    results = {d: s for d, s in zip(result.dims, result.shape) if d != "frames"}

    for dim in results:
        assert expected[dim] == pytest.approx(results[dim], 1)
    return None
