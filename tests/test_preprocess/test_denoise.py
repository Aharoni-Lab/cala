import dataclasses

import cv2
import numpy as np
import pytest
import xarray as xr

from cala.nodes.prep.denoise import Denoiser


class TestStreamingDenoiser:
    @pytest.fixture
    def gaussian_params(self) -> dict:
        """Create default parameters for testing"""
        return {"method": "gaussian", "kwargs": {"ksize": (5, 5), "sigmaX": 1.5}}

    @pytest.fixture
    def median_params(self) -> dict:
        """Create StreamingDenoiser instance with median method"""
        return {"method": "median", "kwargs": {"ksize": 5}}

    @pytest.fixture
    def bilateral_params(self) -> dict:
        """Create StreamingDenoiser instance with bilateral method"""
        return {
            "method": "bilateral",
            "kwargs": {"d": 5, "sigmaColor": 75, "sigmaSpace": 75},
        }

    def test_initialization(self, gaussian_params) -> None:
        """Test proper initialization of StreamingDenoiser"""
        denoiser = Denoiser(gaussian_params)
        assert denoiser.params.method == gaussian_params.method
        assert denoiser.params.kwargs == gaussian_params.kwargs
        assert denoiser._func == cv2.GaussianBlur

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

    def test_different_kernel_sizes(self, gaussian_params, raw_calcium_video: xr.DataArray) -> None:
        """Test denoising with different kernel sizes"""
        video = raw_calcium_video
        frame = video[0]

        kernel_sizes = [(3, 3), (5, 5), (7, 7)]
        for size in kernel_sizes:
            params = dataclasses.replace(gaussian_params, kwargs={"ksize": size, "sigmaX": 1.5})
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
