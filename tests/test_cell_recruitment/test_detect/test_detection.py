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

    def test_max_projection_with_known_input(self):
        # Create a controlled input with known maximum values
        data = np.zeros((5, 10, 10), dtype=np.uint8)
        # Place max values at known positions
        data[0, 5, 5] = 100  # Frame 0
        data[2, 2, 2] = 150  # Frame 2
        data[4, 7, 7] = 200  # Frame 4

        coords = {
            "frames": np.arange(5),
            "width": np.arange(10),
            "height": np.arange(10),
        }
        dims = ("frames", "width", "height")
        data_array = xr.DataArray(data, coords=coords, dims=dims)

        detector = Detector(method="rolling", chunk_size=3, step_size=2)
        max_projections = detector._compute_max_projections(data_array)

        first_projection = np.zeros((10, 10), dtype=np.uint8)
        first_projection[5, 5] = 100
        first_projection[2, 2] = 150

        assert np.all(max_projections.isel(sample=0) == first_projection)

        second_projection = np.zeros((10, 10), dtype=np.uint8)
        second_projection[2, 2] = 150
        second_projection[7, 7] = 200
        assert np.all(max_projections.isel(sample=1) == second_projection)

    def test_local_maxima_with_controlled_input(self):
        # Create a frame with known local maxima
        frame = np.zeros((20, 20), dtype=np.uint8)
        # Create two peaks with different intensities
        frame[5, 5] = 100
        frame[15, 15] = 150
        # Add some noise around the peaks
        frame[5, 6] = 80
        frame[6, 5] = 80
        frame[14, 15] = 100
        frame[15, 14] = 100

        local_maxima = Detector._find_local_maxima(
            frame, k0=2, k1=4, intensity_threshold=10
        )

        # Should detect both peaks
        assert local_maxima[5, 5] == 1
        assert local_maxima[15, 15] == 1
        # Total number of maxima should be 2
        assert np.sum(local_maxima) == 2

    def test_complete_pipeline_with_controlled_input(self):
        # Create a movie with known cell positions
        data = np.zeros((10, 30, 30), dtype=np.uint8)
        # Create two persistent cells
        for frame in range(10):
            # Cell 1 at (10, 10) with intensity 100
            data[frame, 10, 10] = 100
            # Add some noise around cell 1
            data[frame, 10, 11] = 80
            data[frame, 11, 10] = 80

            # Cell 2 at (20, 20) with intensity 150
            data[frame, 20, 20] = 150
            # Add some noise around cell 2
            data[frame, 20, 21] = 100
            data[frame, 21, 20] = 100

        coords = {
            "frames": np.arange(10),
            "width": np.arange(30),
            "height": np.arange(30),
        }
        dims = ("frames", "width", "height")
        data_array = xr.DataArray(data, coords=coords, dims=dims)

        detector = Detector(
            method="rolling",
            chunk_size=5,
            step_size=5,
            local_max_radius=3,
            intensity_threshold=10,
        )

        detector.fit(data_array)
        seeds = detector.transform(data_array)

        # Should find exactly 2 seeds
        assert len(seeds) == 2

        # Convert seeds to set of (width, height) tuples for easy comparison
        seed_positions = {(row.width, row.height) for _, row in seeds.iterrows()}
        expected_positions = {(10, 10), (20, 20)}
        assert seed_positions == expected_positions
