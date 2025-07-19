import dataclasses
from typing import Any

import cv2
import numpy as np
import pytest
import xarray as xr
from scipy.ndimage import uniform_filter
from skimage.morphology import disk

from cala.streaming.nodes.preprocess import (
    BackgroundEraser,
    BackgroundEraserParams,
)


class TestBackgroundEraser:
    @pytest.fixture
    def default_params(self) -> BackgroundEraserParams:
        """Create default parameters for testing"""
        params = BackgroundEraserParams(method="uniform", kernel_size=3)
        return params

    @pytest.fixture
    def eraser_uniform(self, default_params: BackgroundEraserParams) -> BackgroundEraser:
        """Create BackgroundEraser instance with uniform method"""
        return BackgroundEraser(default_params)

    @pytest.fixture
    def eraser_tophat(self, default_params: BackgroundEraserParams) -> BackgroundEraser:
        """Create BackgroundEraser instance with tophat method"""
        params = dataclasses.replace(default_params, method="tophat")
        return BackgroundEraser(params)

    def test_initialization(self, default_params: BackgroundEraserParams) -> None:
        """Test proper initialization of BackgroundEraser"""
        eraser = BackgroundEraser(default_params)
        assert eraser.params.method == default_params.method
        assert eraser.params.kernel_size == default_params.kernel_size

    def test_parameter_validation(self) -> None:
        """Test parameter validation"""
        # Test invalid method
        with pytest.raises(ValueError, match="method must be one of"):
            BackgroundEraser(BackgroundEraserParams(method="invalid"))

        # Test invalid kernel size
        with pytest.raises(ValueError, match="kernel_size must be greater than zero"):
            BackgroundEraser(BackgroundEraserParams(kernel_size=0))

    def test_uniform_background_removal(
        self, eraser_uniform: BackgroundEraser, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test background removal using uniform method"""
        video = raw_calcium_video
        frame = video[0]

        # Process frame
        result = eraser_uniform.transform_one(frame)

        # Check basic properties
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

        # Verify background removal
        background = uniform_filter(frame, size=eraser_uniform.params.kernel_size)
        expected = frame - background
        expected.values[expected < 0] = 0

        np.testing.assert_array_almost_equal(result, expected)

    def test_tophat_background_removal(
        self, eraser_tophat: BackgroundEraser, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test background removal using tophat method"""
        video = raw_calcium_video
        frame = video[0]

        # Process frame
        result = eraser_tophat.transform_one(frame)

        # Check basic properties
        assert result.shape == frame.shape
        assert result.dtype == np.float32
        assert not np.any(np.isnan(result))

        # Verify background removal
        kernel = disk(eraser_tophat.params.kernel_size)
        expected = cv2.morphologyEx(
            frame.values.astype(np.float32), cv2.MORPH_TOPHAT, kernel.astype(np.uint8)
        )

        np.testing.assert_array_almost_equal(result, expected)

    def test_streaming_consistency(
        self, eraser_uniform: BackgroundEraser, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test consistency of streaming background removal"""
        video = raw_calcium_video
        frames = [video[i] for i in range(5)]

        # Process frames sequentially
        streaming_results = []
        for frame in frames:
            eraser_uniform.learn_one(frame)  # Should be a no-op
            streaming_results.append(eraser_uniform.transform_one(frame))

        # Process frames in batch
        batch_results = []
        for frame in frames:
            background = uniform_filter(frame.values, size=eraser_uniform.params.kernel_size)
            result = frame.values - background
            result[result < 0] = 0
            batch_results.append(result)

        # Compare results
        for streaming, batch in zip(streaming_results, batch_results):
            np.testing.assert_array_almost_equal(streaming, batch)

    def test_different_kernel_sizes(
        self, default_params: Any, raw_calcium_video: xr.DataArray
    ) -> None:
        """Test background removal with different kernel sizes"""
        video = raw_calcium_video
        frame = video[0]

        kernel_start = 100
        kernel_jump = 25
        kernel_sizes = range(kernel_start, 200, kernel_jump)
        for size in kernel_sizes:
            params = dataclasses.replace(default_params)
            params.kernel_size = size
            eraser = BackgroundEraser(params)

            result = eraser.transform_one(frame)
            assert result.shape == frame.shape

            # Larger kernels should remove less background
            if size > kernel_start:
                prev_params = dataclasses.replace(params)
                prev_params.kernel_size = size - kernel_jump
                prev_eraser = BackgroundEraser(prev_params)
                prev_result = prev_eraser.transform_one(frame)
                # plt.imsave(
                #     f"{size - kernel_jump}_result.png",
                #     prev_result.values.astype(np.float32),
                # )
                # plt.imsave(
                #     f"{size}_result.png",
                #     result.values.astype(np.float32),
                # )
                assert np.mean(result.values) > np.mean(prev_result.values)

    def test_edge_cases(self, default_params: Any) -> None:
        """Test handling of edge cases"""
        # Test constant frame
        constant_frame = xr.DataArray(np.ones((50, 50)))
        eraser = BackgroundEraser(default_params)
        result = eraser.transform_one(constant_frame)
        assert np.allclose(result, 0)  # Background should be removed

        # Test zero frame
        zero_frame = xr.DataArray(np.zeros((50, 50)))
        result = eraser.transform_one(zero_frame)
        assert np.allclose(result, 0)
