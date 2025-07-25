import numpy as np
import pytest
import xarray as xr

from cala.util.buffer import Buffer


@pytest.fixture
def frame_shape():
    return 512, 512


@pytest.fixture
def buffer_size():
    return 5


@pytest.fixture
def sample_frame(frame_shape):
    # Create an xarray DataArray instead of a numpy array
    data = np.ones(frame_shape, dtype=np.uint8) * 128
    return xr.DataArray(data)


@pytest.fixture
def buffer(buffer_size):
    return Buffer(size=buffer_size)


def test_buffer_initialization(buffer, buffer_size):
    """Test if Buffer is initialized correctly with given parameters."""
    assert buffer.size == buffer_size
    assert len(buffer.frames) == 0


def test_add_frame(buffer, sample_frame):
    """Test adding a single frame to the buffer."""
    buffer.add(sample_frame)
    assert len(buffer.frames) == 1
    # Check that the frame in the buffer equals the sample frame
    assert np.array_equal(buffer.get_latest(), sample_frame)


def test_add_multiple_frames(buffer, sample_frame, buffer_size):
    """Test adding multiple frames and checking buffer size."""
    # Add more frames than buffer size to test deque behavior
    num_frames = buffer_size + 2

    for i in range(num_frames):
        # Create a new frame with different values for each iteration
        data = np.ones(sample_frame.shape, dtype=np.uint8) * (i + 1)
        frame = xr.DataArray(data)
        buffer.add(frame)

    # Check if buffer doesn't exceed buffer_size
    assert len(buffer.frames) == buffer_size

    # Verify the most recent frame's values
    expected_value = num_frames
    last_frame = buffer.get_latest(1)
    assert np.all(last_frame.values.flatten()[0] == expected_value)


def test_get_frame_not_enough_frames(buffer):
    """Test getting frames when buffer doesn't have enough raises error."""
    with pytest.raises(ValueError, match="Buffer does not have enough frames"):
        buffer.get_latest(1)


def test_is_ready(buffer, sample_frame):
    """Test is_ready method."""
    assert not buffer.is_ready(1)

    buffer.add(sample_frame)
    assert buffer.is_ready(1)
    assert not buffer.is_ready(2)

    buffer.add(sample_frame)
    assert buffer.is_ready(2)


def test_get_multiple_frames(buffer, sample_frame):
    """Test getting multiple frames from the buffer."""
    # Add frames with different values
    for i in range(3):
        data = np.ones(sample_frame.shape, dtype=np.uint8) * (i + 1)
        frame = xr.DataArray(data)
        buffer.add(frame)

    # Get the last 2 frames
    frames = buffer.get_latest(2)

    # Check shape and dimensions
    assert frames.shape[0] == 2
    assert frames.shape[1:] == sample_frame.shape

    # Check values (last frame should be 3, second-to-last should be 2)
    assert np.all(frames.values[1].flatten()[0] == 3)
    assert np.all(frames.values[0].flatten()[0] == 2)


def test_buffer_wrapping(buffer, sample_frame, buffer_size):
    """Test buffer wrapping behavior when buffer is full."""
    # Fill buffer
    for i in range(buffer_size):
        data = np.ones(sample_frame.shape, dtype=np.uint8) * (i + 1)
        frame = xr.DataArray(data)
        buffer.add(frame)

    # Add one more frame to trigger wrapping
    new_value = buffer_size + 1
    data = np.ones(sample_frame.shape, dtype=np.uint8) * new_value
    new_frame = xr.DataArray(data)
    buffer.add(new_frame)

    # Check buffer size remains at max
    assert len(buffer.frames) == buffer_size

    # Get all frames
    all_frames = buffer.get_latest(buffer_size)

    # Check the oldest frame is now the second one (value=2)
    # and newest is the last added (value=buffer_size+1)
    assert np.all(all_frames.values[0].flatten()[0] == 2)
    assert np.all(all_frames.values[-1].flatten()[0] == new_value)
