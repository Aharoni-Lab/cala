from collections import deque

import xarray as xr


class Buffer:
    def __init__(self, buffer_size, frame_shape):
        """
        Initialize the ring buffer with:
          - buffer_size: number of frames to store
          - frame_shape: the shape of each frame (e.g. (height, width, channels))
          - dtype: data type of the frames (e.g. np.uint8 for typical video data)
        """
        self.buffer_size = buffer_size
        self.frame_shape = frame_shape

        self.buffer: deque[xr.DataArray] = deque()

    def add_frame(self, frame: xr.DataArray):
        """
        Add a new frame to the ring buffer.
        The frame must match the frame_shape and dtype specified in the constructor.
        """
        if frame.shape != self.frame_shape:
            raise ValueError("Frame shape does not match buffer frame_shape.")

        self.buffer.append(frame)
        if len(self.buffer) > self.buffer_size:
            self.buffer.popleft()

    def get_frame(self, n: int = 1) -> xr.DataArray:
        """Get n most recent frames.

        Returns:
            xr.DataArray: A 3D array containing the stacked frames.
        """
        if not self.is_ready(n):
            raise ValueError("Buffer does not have enough frames.")

        if n == 1:
            return self.buffer[0]

        return xr.concat(list(self.buffer)[:n], dim="frame")

    def is_ready(self, num_frames: int) -> bool:
        """Check if buffer has enough frames."""
        return len(self.buffer) >= num_frames
