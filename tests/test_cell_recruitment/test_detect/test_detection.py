from unittest.mock import patch

import numpy as np
import pytest
import xarray as xr

from cala.cell_recruitment.detect import Detector


class TestDetector:
    @pytest.fixture
    def sample_data(self):
        # Create a sample xarray DataArray with random data
        data = np.random.randint(0, 256, size=(10, 100, 100), dtype=np.uint8)
        coords = {
            "frames": np.arange(10),
            "width": np.arange(100),
            "height": np.arange(100),
        }
        dims = ("frames", "width", "height")
        return xr.DataArray(data, coords=coords, dims=dims)

    def test_fit_without_errors(self, sample_data):
        detector = Detector()
        # Ensure that fit method runs without errors
        detector.fit(sample_data)
        assert detector.max_projection_ is not None

    def test_transform_without_fit(self, sample_data):
        detector = Detector()
        with pytest.raises(ValueError) as excinfo:
            detector.transform(sample_data)
        assert "Fit method must be run before transform" in str(excinfo.value)

    def test_transform_after_fit(self, sample_data):
        detector = Detector()
        detector.fit(sample_data)
        seeds = detector.transform(sample_data)
        # Check that seeds is a DataFrame and has expected columns
        assert "width" in seeds.columns
        assert "height" in seeds.columns
        assert "seeds" in seeds.columns

    def test_method_parameter_validation(self):
        with pytest.raises(ValueError) as excinfo:
            Detector(method="invalid_method")
        assert "Method must be either 'rolling' or 'random'" in str(excinfo.value)

    def test_compute_max_projections_rolling(self, sample_data):
        detector = Detector(method="rolling", chunk_size=5, step_size=2)
        max_projections = detector._compute_max_projections(sample_data)
        # Expect number of samples based on chunking
        expected_samples = np.ceil((10 - 5) / 2) + 1
        assert len(max_projections.coords["sample"]) == expected_samples

    def test_compute_max_projections_random(self, sample_data):
        detector = Detector(method="random", chunk_size=5, num_chunks=3)
        max_projections = detector._compute_max_projections(sample_data)
        assert len(max_projections.coords["sample"]) == 3

    def test_get_max_indices_rolling(self):
        detector = Detector(method="rolling", chunk_size=5, step_size=2)
        indices = detector._get_max_indices(10)
        expected_num_indices = np.ceil((10 - 5) / 2) + 1
        assert len(indices) == expected_num_indices

    def test_get_max_indices_random(self):
        detector = Detector(method="random", chunk_size=5, num_chunks=3)
        indices = detector._get_max_indices(10)
        assert len(indices) == 3

    @patch("cala.cell_recruitment.detect.Detector._find_local_maxima")
    def test_find_local_maxima_called(self, mock_find_local_maxima, sample_data):
        # Mock the _find_local_maxima method
        mock_find_local_maxima.return_value = np.zeros((100, 100), dtype=np.uint8)
        detector = Detector()
        detector.fit(sample_data)
        detector.transform(sample_data)
        assert mock_find_local_maxima.called

    def test_find_local_maxima_output(self):
        # Test the static method directly with a sample frame
        frame = np.random.randint(0, 256, size=(100, 100), dtype=np.uint8)
        local_maxima = Detector._find_local_maxima(
            frame, k0=2, k1=5, intensity_threshold=10
        )
        # Check that output is a binary image
        assert local_maxima.dtype == np.uint8
        assert set(np.unique(local_maxima)) <= {0, 1}

    def test_empty_input(self):
        # Test with an empty DataArray
        data = np.empty((0, 100, 100), dtype=np.uint8)
        coords = {
            "frames": np.arange(0),
            "width": np.arange(100),
            "height": np.arange(100),
        }
        dims = ("frames", "width", "height")
        empty_data = xr.DataArray(data, coords=coords, dims=dims)
        detector = Detector()
        with pytest.raises(ValueError):
            detector.fit(empty_data)

    def test_large_kernel_size(self, sample_data):
        detector = Detector(local_max_radius=50)
        detector.fit(sample_data)
        seeds = detector.transform(sample_data)
        # Check that seeds are still computed
        assert not seeds.empty

    def test_intensity_threshold(self, sample_data):
        detector = Detector(intensity_threshold=500)  # High threshold
        detector.fit(sample_data)
        seeds = detector.transform(sample_data)
        # Expect no seeds due to out of bounds threshold
        assert len(seeds) == 0

    def test_iter_axis_modification(self):
        data = np.random.randint(0, 256, size=(10, 100, 100), dtype=np.uint8)
        coords = {
            "time": np.arange(10),
            "width": np.arange(100),
            "height": np.arange(100),
        }
        dims = ("time", "width", "height")
        data_array = xr.DataArray(data, coords=coords, dims=dims)
        detector = Detector(iter_axis="time")
        detector.fit(data_array)
        seeds = detector.transform(data_array)
        assert not seeds.empty
