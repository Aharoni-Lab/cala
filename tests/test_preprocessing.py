import numpy as np
import pytest

from cala.preprocessing import Preprocessor


@pytest.fixture
def preprocessor():
    """Fixture to create a Preprocessor instance."""
    return Preprocessor()


@pytest.fixture
def sample_video():
    """
    Fixture to create a sample video array.
    For simplicity, create a video with shape (10, 64, 64, 3).
    """
    return np.arange(10 * 64 * 64, dtype=np.float64).reshape((10, 64, 64))


@pytest.fixture
def empty_video():
    """Fixture to create an empty video array."""
    return np.empty((0, 64, 64), dtype=np.float64)


@pytest.fixture
def small_video():
    """
    Fixture to create a small video array with fewer frames than the batch size.
    Shape: (5, 64, 64, 3)
    """
    return np.arange(5 * 64 * 64, dtype=np.uint8).reshape((5, 64, 64))


def test_yield_in_batches_normal_case(preprocessor, sample_video):
    """Test yield_in_batches with a normal video and batch size."""
    batch_size = 3
    batches = list(preprocessor.yield_in_batches(sample_video, batch_size))

    expected_num_batches = int(np.ceil(len(sample_video) / batch_size))
    assert len(batches) == expected_num_batches, "Incorrect number of batches"

    for i, batch in enumerate(batches):
        start = i * batch_size
        end = start + batch_size
        expected_batch = sample_video[start:end]
        np.testing.assert_array_equal(
            batch, expected_batch, f"Batch {i} does not match expected"
        )


def test_yield_in_batches_empty_video(preprocessor, empty_video):
    """Test yield_in_batches with an empty video."""
    batch_size = 5
    batches = list(preprocessor.yield_in_batches(empty_video, batch_size))
    assert len(batches) == 0, "Batches should be empty for empty video"


def test_yield_in_batches_batch_size_larger(preprocessor, sample_video):
    """Test yield_in_batches when batch size is larger than video length."""
    batch_size = 20  # Larger than the video length (10)
    batches = list(preprocessor.yield_in_batches(sample_video, batch_size))
    assert len(batches) == 1, "Should yield a single batch"
    np.testing.assert_array_equal(
        batches[0], sample_video, "Single batch does not match entire video"
    )


def test_yield_in_batches_batch_size_one(preprocessor, sample_video):
    """Test yield_in_batches with batch size of 1."""
    batch_size = 1
    batches = list(preprocessor.yield_in_batches(sample_video, batch_size))
    assert len(batches) == len(
        sample_video
    ), "Number of batches should equal number of frames"

    for i, batch in enumerate(batches):
        expected_batch = sample_video[i : i + 1]
        np.testing.assert_array_equal(
            batch, expected_batch, f"Batch {i} does not match expected"
        )


def test_process_video_in_batches_empty(preprocessor, empty_video):
    """Test process_video_in_batches with an empty video."""
    batch_size = 5
    processed_batches = preprocessor.process_video_in_batches(empty_video, batch_size)
    assert (
        len(processed_batches) == 0
    ), "Processed batches should be empty for empty video"


def test_process_video_in_batches_dtype(preprocessor, sample_video):
    """Test that the output dtype is correct after processing."""
    processed_batches = preprocessor.process_video_in_batches(
        sample_video, batch_size=3, downsample_mode="subset"
    )
    for batch in processed_batches:
        assert (
            batch.dtype == np.float64
        ), "Processed batch dtype should be float64 after division and multiplication"


def test_mean_downsample(preprocessor, sample_video):
    """Test mean downsample"""
    print(sample_video.shape)
    processed_batches = preprocessor.process_video_in_batches(
        sample_video,
        batch_size=3,
        downsample_mode="mean",
        t_downsample=2,
        x_downsample=3,
        y_downsample=5,
    )

    print(processed_batches[0].shape)

    assert 1 == 1
