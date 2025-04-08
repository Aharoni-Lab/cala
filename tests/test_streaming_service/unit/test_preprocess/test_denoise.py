import dataclasses

import cv2
import numpy as np
import pytest
import xarray as xr

from cala.streaming.nodes.preprocess import (
    Denoiser,
    DenoiserParams,
)


class TestStreamingDenoiser:
    @pytest.fixture
    def default_params(self) -> DenoiserParams:
        """Create default parameters for testing"""
        params = DenoiserParams(method="gaussian", kwargs={"ksize": (5, 5), "sigmaX": 1.5})
        return params

    @pytest.fixture
    def denoiser_gaussian(self, default_params: DenoiserParams) -> Denoiser:
        """Create StreamingDenoiser instance with Gaussian method"""
        return Denoiser(default_params)

    @pytest.fixture
    def denoiser_median(self, default_params: DenoiserParams) -> Denoiser:
        """Create StreamingDenoiser instance with median method"""
        params = dataclasses.replace(default_params, method="median", kwargs={"ksize": 5})
        return Denoiser(params)

    @pytest.fixture
    def denoiser_bilateral(self, default_params: DenoiserParams) -> Denoiser:
        """Create StreamingDenoiser instance with bilateral method"""
        params = dataclasses.replace(
            default_params,
            method="bilateral",
            kwargs={"d": 5, "sigmaColor": 75, "sigmaSpace": 75},
        )
        return Denoiser(params)

    def test_initialization(self, default_params: DenoiserParams) -> None:
        """Test proper initialization of StreamingDenoiser"""
        denoiser = Denoiser(default_params)
        assert denoiser.params.method == default_params.method
        assert denoiser.params.kwargs == default_params.kwargs
        assert denoiser.func == cv2.GaussianBlur

    def test_parameter_validation(self) -> None:
        """Test parameter validation"""
        # Test invalid method
        with pytest.raises(ValueError, match="denoise method .* not understood"):
            Denoiser(DenoiserParams(method="invalid"))

    def test_gaussian_denoising(
        self, denoiser_gaussian: Denoiser, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test denoising using Gaussian method"""
        video = raw_calcium_video
        frame = video[0]

        # Process frame
        result = denoiser_gaussian.transform_one(frame)

        # Check basic properties
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

        # Verify denoising
        expected = cv2.GaussianBlur(
            frame.values.astype(np.float32), **denoiser_gaussian.params.kwargs
        )

        np.testing.assert_array_almost_equal(result, expected)

    def test_median_denoising(
        self, denoiser_median: Denoiser, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test denoising using median method"""
        video = raw_calcium_video
        frame = video[0]

        # Process frame
        result = denoiser_median.transform_one(frame)

        # Check basic properties
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

        # Verify denoising
        expected = cv2.medianBlur(frame.values.astype(np.float32), **denoiser_median.params.kwargs)

        np.testing.assert_array_almost_equal(result, expected)

    def test_bilateral_denoising(
        self, denoiser_bilateral: Denoiser, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test denoising using bilateral method"""
        video = raw_calcium_video
        frame = video[0]

        # Process frame
        result = denoiser_bilateral.transform_one(frame)

        # Check basic properties
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

        # Verify denoising
        expected = cv2.bilateralFilter(
            frame.values.astype(np.float32), **denoiser_bilateral.params.kwargs
        )

        np.testing.assert_array_almost_equal(result, expected)

    def test_streaming_consistency(
        self, denoiser_gaussian: Denoiser, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test consistency of streaming denoising"""
        video = raw_calcium_video
        frames = [video[i] for i in range(5)]

        # Process frames sequentially
        streaming_results = []
        for frame in frames:
            denoiser_gaussian.learn_one(frame)  # Should be a no-op
            streaming_results.append(denoiser_gaussian.transform_one(frame))

        # Process frames in batch
        batch_results = []
        for frame in frames:
            result = cv2.GaussianBlur(
                frame.values.astype(np.float32), **denoiser_gaussian.params.kwargs
            )
            batch_results.append(result)

        # Compare results
        for streaming, batch in zip(streaming_results, batch_results):
            np.testing.assert_array_almost_equal(streaming, batch)

    def test_different_kernel_sizes(
        self, default_params: DenoiserParams, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test denoising with different kernel sizes"""
        video = raw_calcium_video
        frame = video[0]

        kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        for size in kernel_sizes:
            params = dataclasses.replace(default_params, kwargs={"ksize": size, "sigmaX": 1.5})
            denoiser = Denoiser(params)

            result = denoiser.transform_one(frame)
            assert result.shape == frame.shape

            # Larger kernels should produce more smoothing
            if size[0] > 3:
                prev_params = dataclasses.replace(
                    params, kwargs={"ksize": (size[0] - 2, size[1] - 2), "sigmaX": 1.5}
                )
                prev_denoiser = Denoiser(prev_params)
                prev_result = prev_denoiser.transform_one(frame)
                assert np.std(result) < np.std(prev_result)

    def test_edge_cases(self, default_params: DenoiserParams) -> None:
        """Test handling of edge cases"""
        # Test constant frame
        constant_frame = xr.DataArray(np.ones((50, 50)))
        denoiser = Denoiser(default_params)
        result = denoiser.transform_one(constant_frame)
        assert np.allclose(result, 1)  # Should preserve constant values

        # Test zero frame
        zero_frame = xr.DataArray(np.zeros((50, 50)))
        result = denoiser.transform_one(zero_frame)
        assert np.allclose(result, 0)  # Should preserve zero values

    def test_get_info(self, denoiser_gaussian: Denoiser) -> None:
        """Test get_info method"""
        info = denoiser_gaussian.get_info()
        assert info["method"] == denoiser_gaussian.params.method
        assert info["func"] == "GaussianBlur"
        assert info["kwargs"] == denoiser_gaussian.params.kwargs
