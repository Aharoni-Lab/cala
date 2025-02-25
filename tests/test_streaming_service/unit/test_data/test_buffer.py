import numpy as np
import pytest

from cala.streaming.data.buffer import RingBuffer


@pytest.fixture
def frame_shape():
    return (512, 512, 3)


@pytest.fixture
def buffer_size():
    return 5


@pytest.fixture
def sample_frame(frame_shape):
    return np.ones(frame_shape, dtype=np.uint8) * 128


@pytest.fixture
def ring_buffer(buffer_size, frame_shape):
    return RingBuffer(buffer_size=buffer_size, frame_shape=frame_shape)


def test_ring_buffer_initialization(ring_buffer, buffer_size, frame_shape):
    """Test if RingBuffer is initialized correctly with given parameters."""
    assert ring_buffer.buffer_size == buffer_size
    assert ring_buffer.frame_shape == frame_shape
    assert ring_buffer.dtype == np.uint8
    assert ring_buffer.index == 0
    assert ring_buffer.buffer.shape == (buffer_size, *frame_shape)
    assert np.all(ring_buffer.buffer == 0)


def test_add_frame(ring_buffer, sample_frame):
    """Test adding a single frame to the buffer."""
    ring_buffer.add_frame(sample_frame)
    assert ring_buffer.index == 1
    assert np.array_equal(ring_buffer.get_frame(0), sample_frame)


def test_add_multiple_frames(ring_buffer, sample_frame, buffer_size):
    """Test adding multiple frames and wrapping behavior."""
    # Add more frames than buffer size to test wrapping
    num_frames = buffer_size + 2

    for i in range(num_frames):
        frame = sample_frame * (i + 1)
        ring_buffer.add_frame(frame)

    # Check if the index wrapped correctly
    assert ring_buffer.index == 2

    # Verify the most recent frame
    assert np.array_equal(ring_buffer.get_frame(0), sample_frame * num_frames)


def test_get_frame_invalid_index(ring_buffer):
    """Test getting a frame with invalid index raises error."""
    with pytest.raises(IndexError):
        ring_buffer.get_frame(ring_buffer.buffer_size)


def test_add_frame_wrong_shape(ring_buffer):
    """Test adding a frame with wrong shape raises error."""
    wrong_shape_frame = np.ones((16, 16, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Frame shape does not match"):
        ring_buffer.add_frame(wrong_shape_frame)


def test_add_frame_wrong_dtype(ring_buffer, frame_shape):
    """Test adding a frame with wrong dtype raises error."""
    wrong_dtype_frame = np.ones(frame_shape, dtype=np.float32)
    with pytest.raises(ValueError, match="Frame dtype does not match"):
        ring_buffer.add_frame(wrong_dtype_frame)


@pytest.mark.parametrize(
    "num_frames,scenario",
    [
        (0, "empty"),  # No frames added
        (3, "partially filled"),  # buffer_size - 2
        (5, "exactly filled"),  # buffer_size
        (7, "wrapped once"),  # buffer_size + 2
        (10, "wrapped multiple"),  # buffer_size * 2
    ],
)
def test_get_buffer_in_order(
    ring_buffer, sample_frame, buffer_size, num_frames, scenario
):
    """Test getting buffer frames in chronological order from different wrap-around points."""
    # Reset buffer for clean state
    ring_buffer = RingBuffer(buffer_size=buffer_size, frame_shape=sample_frame.shape)

    # Add frames with different values
    for i in range(num_frames):
        frame = sample_frame * (i + 1)
        ring_buffer.add_frame(frame)

    ordered_buffer = ring_buffer.get_buffer_in_order()

    # Calculate expected frames based on how many frames were added
    if num_frames == 0:
        assert (
            len(ordered_buffer) == 0
        ), f"Empty buffer should return empty array in {scenario} case"
        return

    start_idx = max(0, num_frames - buffer_size)
    expected_frames = [sample_frame * (i + 1) for i in range(start_idx, num_frames)]
    expected_frames = np.stack(expected_frames)

    # Verify the order and values
    assert ordered_buffer.shape[0] == min(
        buffer_size, num_frames
    ), f"Buffer size mismatch in {scenario} case"
    assert np.array_equal(
        ordered_buffer, expected_frames
    ), f"Frame content mismatch in {scenario} case"

    # Additional verification of current buffer state
    # Verify most recent frame is accessible via get_frame(0)
    assert np.array_equal(
        ring_buffer.get_frame(0), sample_frame * num_frames
    ), f"Most recent frame mismatch in {scenario} case"


def test_get_all_frames(ring_buffer, sample_frame, buffer_size):
    """Test getting all frames from the buffer."""
    # Add frames with different values
    for i in range(buffer_size):
        frame = sample_frame * (i + 1)
        ring_buffer.add_frame(frame)

    all_frames = ring_buffer.get_all_frames()

    # Verify it's a copy
    assert all_frames is not ring_buffer.buffer

    # Verify contents
    assert np.array_equal(all_frames, ring_buffer.buffer)


def test_frame_retrieval_order(ring_buffer, sample_frame, buffer_size):
    """Test retrieving frames in reverse chronological order."""
    # Add frames with different values
    for i in range(buffer_size):
        frame = sample_frame * (i + 1)
        ring_buffer.add_frame(frame)

    # Verify frames can be retrieved in reverse chronological order
    for i in range(buffer_size):
        expected_frame = sample_frame * (buffer_size - i)
        assert np.array_equal(ring_buffer.get_frame(i), expected_frame)
