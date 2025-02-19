import numpy as np


class RingBuffer:
    def __init__(self, buffer_size, frame_shape, dtype=np.uint8):
        """
        Initialize the ring buffer with:
          - buffer_size: number of frames to store
          - frame_shape: the shape of each frame (e.g. (height, width, channels))
          - dtype: data type of the frames (e.g. np.uint8 for typical video data)
        """
        self.buffer_size = buffer_size
        self.frame_shape = frame_shape
        self.dtype = dtype

        # pre-allocate the buffer as a 3D array: (buffer_size, height, width)
        self.buffer = np.zeros((buffer_size, *frame_shape), dtype=dtype)

        # keep track of where the next frame will be inserted
        self.index = 0

    def add_frame(self, frame: np.ndarray):
        """
        Add a new frame to the ring buffer.
        The frame must match the frame_shape and dtype specified in the constructor.
        """
        if frame.shape != self.frame_shape:
            raise ValueError("Frame shape does not match ring buffer frame_shape.")
        if frame.dtype != self.dtype:
            raise ValueError("Frame dtype does not match ring buffer dtype.")

        self.buffer[self.index] = frame

        # Increment the index and wrap using modulus
        self.index = (self.index + 1) % self.buffer_size

    def get_frame(self, idx_from_latest=0) -> np.ndarray:
        """
        Retrieve a frame from the ring buffer.

        By default, idx_from_latest=0 means the most recently added frame.
        idx_from_latest=1 means the frame before that, and so on.

        For example, if `self.index == 10`, then the most recent frame
        is at index 9 (because after the last add, self.index points
        to the *next* position).
        """
        if idx_from_latest >= self.buffer_size:
            raise IndexError(
                "idx_from_latest is out of range for the current buffer size."
            )

        # Compute the actual index in the buffer
        actual_index = (self.index - 1 - idx_from_latest) % self.buffer_size
        return self.buffer[actual_index]

    def get_all_frames(self) -> np.ndarray:
        """
        Returns the entire buffer. Note that the frames may not be
        in the order they were added if self.index has wrapped around.
        You can reorder them if you need chronological order.
        """
        return self.buffer.copy()

    def get_buffer_in_order(self) -> np.ndarray:
        """
        Return the buffer frames in chronological order (oldest to newest).
        This is useful if you want a sequence of frames in the exact order
        they were written, up to the current index.
        """
        # The newest frame is just before self.index,
        # the oldest frame is at self.index in circular sense.
        # We'll construct a new array in chronological order.

        # Partition the buffer into two slices:
        # 1. from self.index to end
        # 2. from 0 to self.index - 1
        return np.concatenate(
            (self.buffer[self.index:], self.buffer[: self.index]), axis=0
        )
