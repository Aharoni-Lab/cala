from abc import abstractmethod
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
from numpy.typing import NDArray
from skimage import io


class Stream(Protocol):
    """Protocol defining the interface for video streams."""

    @abstractmethod
    def __iter__(self) -> Iterator[NDArray]:
        """Iterate over frames."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close the stream and release resources."""
        ...


class OpenCVStream(Stream):
    """OpenCV-based implementation of video streaming."""

    def __init__(self, video_path: Path | str) -> None:
        video_path = Path(video_path)

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

    def __iter__(self) -> Iterator[NDArray]:
        """
        Yields:
            NDArray: Next frame from the video
        """
        while self._cap.isOpened():
            ret, frame = self._cap.read()
            if not ret:
                break
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            yield frame

    def close(self) -> None:
        """Close the video stream and release resources."""
        if self._cap is not None:
            self._cap.release()


class ImageStream(Stream):
    """Stream implementation for sequence of TIFF files."""

    def __init__(self, files: list[Path]) -> None:
        self._files = files

        # Validate first file to ensure it's readable and get sample shape
        frame = io.imread(self._files[0])
        if len(frame.shape) != 2:
            raise ValueError("TIFF files must be grayscale")
        self._sample_shape = frame.shape

    def __iter__(self) -> Iterator[NDArray]:
        for file in self._files:
            frame = io.imread(file)
            if len(frame.shape) != 2:
                raise ValueError(f"File {file} is not grayscale")
            if frame.shape != self._sample_shape:
                raise ValueError(
                    f"Inconsistent frame shape in {file}: "
                    f"expected {self._sample_shape}, got {frame.shape}"
                )
            yield frame

    def close(self) -> None:
        """No resources to close for image sequence."""
        pass


class VideoStream(Stream):
    """Handles streaming from multiple video files sequentially."""

    def __init__(self, video_paths: list[Path]) -> None:
        self._video_paths = video_paths
        self._current_stream: OpenCVStream | None = None

    def __iter__(self) -> Iterator[NDArray]:
        """
        Iterate over frames from all videos sequentially.

        Yields:
            NDArray: Next frame from the current video
        """
        for video_path in self._video_paths:
            self._current_stream = OpenCVStream(video_path)
            yield from self._current_stream
            self._current_stream.close()

    def close(self) -> None:
        """Close the current video stream if open."""
        if self._current_stream is not None:
            self._current_stream.close()


def stream(files: list[str | Path]) -> Stream:
    """
    Create a video stream from the provided video files.

    Args:
        files: List of file paths

    Returns:
        Stream: A stream that yields video frames
    """
    file_paths = [Path(f) if isinstance(f, str) else f for f in files]
    suffix = {path.suffix.lower() for path in file_paths}

    image_format = {".tif", ".tiff"}
    video_format = {".mp4", ".avi", ".webm"}

    if suffix.issubset(video_format):
        return VideoStream(files)
    elif suffix.issubset(image_format):
        return ImageStream(files)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
