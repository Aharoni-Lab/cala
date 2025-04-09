import os
from pathlib import Path

import numpy as np
import pytest

from cala.io import IO, TiffStream


@pytest.fixture
def sample_tiff_dir() -> Path:
    """Fixture providing path to sample TIFF files."""
    base_dir = Path(__file__).resolve().parent
    return base_dir / "fixtures/sample_movies/neurofinder.00.00/images"


@pytest.fixture
def sample_tiff_files(sample_tiff_dir: Path) -> list[Path]:
    """Fixture providing sorted list of sample TIFF files."""
    return sorted(list(sample_tiff_dir.glob("*.tiff")))


@pytest.mark.skipif(os.getenv("CI") == "true", reason="TIFFs unavailable on CI")
def test_tiff_stream_initialization(sample_tiff_files):
    """Test TiffStream initialization with sample files."""
    stream = TiffStream(sample_tiff_files)
    assert stream is not None

    # Should raise error for empty list
    with pytest.raises(ValueError, match="No TIFF files provided"):
        TiffStream([])


@pytest.mark.skipif(os.getenv("CI") == "true", reason="TIFFs unavailable on CI")
def test_tiff_stream_frame_properties(sample_tiff_files):
    """Test frame properties without loading all frames."""
    stream = TiffStream(sample_tiff_files)

    # Get first frame to check properties
    first_frame = next(iter(stream))

    # Check frame properties
    assert isinstance(first_frame, np.ndarray)
    assert first_frame.dtype == np.uint16  # Assuming 16-bit grayscale
    assert len(first_frame.shape) == 2  # Ensure grayscale
    assert not np.any(np.isnan(first_frame))  # No NaN values
    assert np.min(first_frame) >= 0  # Valid pixel values

    # Store shape for comparison
    expected_shape = first_frame.shape

    # Check a few more frames (e.g., every 1000th frame)
    for i, frame in enumerate(stream):
        if i % 1000 == 0:
            assert frame.shape == expected_shape
            assert len(frame.shape) == 2
            assert frame.dtype == np.uint16
            assert not np.any(np.isnan(frame))


@pytest.mark.skipif(os.getenv("CI") == "true", reason="TIFFs unavailable on CI")
def test_io_tiff_handling(sample_tiff_files):
    """Test IO class handles TIFF files correctly."""
    io = IO()

    # Test with TIFF files
    stream = io.stream(sample_tiff_files)
    assert isinstance(stream, TiffStream)

    # Get first frame and verify it's grayscale
    first_frame = next(iter(stream))
    assert len(first_frame.shape) == 2

    # Count frames
    frame_count = 0
    for _ in stream:
        frame_count += 1

    assert frame_count == len(sample_tiff_files)


@pytest.mark.skipif(os.getenv("CI") == "true", reason="TIFFs unavailable on CI")
def test_tiff_stream_cleanup(sample_tiff_files):
    """Test TiffStream cleanup."""
    stream = TiffStream(sample_tiff_files)

    # Stream just one frame
    next(iter(stream))

    # Close should work without errors
    stream.close()


def test_io_mixed_files_error():
    """Test IO raises error for mixed file types."""
    io = IO()

    with pytest.raises(ValueError, match="Cannot mix video and TIFF files"):
        io.stream(["video.mp4", "image.tiff"])


def test_tiff_stream_invalid_file():
    """Test TiffStream handles invalid files appropriately."""
    with pytest.raises(ValueError):
        TiffStream([Path("nonexistent.tiff")])


@pytest.mark.skipif(os.getenv("CI") == "true", reason="TIFFs unavailable on CI")
def test_tiff_stream_random_access(sample_tiff_files):
    """Test accessing random frames in the sequence."""
    # Test a few random frames without loading everything
    test_indices = [0, 100, 1000, 2000]  # Adjust based on your dataset

    for idx in test_indices:
        if idx < len(sample_tiff_files):
            stream = TiffStream([sample_tiff_files[idx]])
            frame = next(iter(stream))
            assert isinstance(frame, np.ndarray)
            assert len(frame.shape) == 2
            stream.close()
