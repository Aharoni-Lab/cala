from abc import abstractmethod
from pathlib import Path
from typing import Protocol, Generator

import cv2
from numpy.typing import NDArray
from skimage import io

from cala.config import config


class Stream(Protocol):
    """Protocol defining the interface for video streams."""

    @abstractmethod
    def __iter__(self) -> Generator[NDArray]:
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
        if not video_path.is_absolute():
            video_path = config.video_dir / video_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

    def __iter__(self) -> Generator[NDArray]:
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


def stream_images(files: list[Path]) -> Generator[NDArray]:
    """Stream implementation for sequence of TIFF files."""
    for file in files:
        frame = io.imread(file)
        if len(frame.shape) != 2:
            raise ValueError(f"File {file} is not grayscale")
        yield frame


def stream_videos(video_paths: list[Path]) -> Generator[NDArray]:
    """
    Iterate over frames from all videos sequentially.

    Yields:
        NDArray: Next frame from the current video
    """
    for video_path in video_paths:
        current_stream = OpenCVStream(video_path)
        yield from current_stream
        current_stream.close()


def stream(files: list[str | Path]) -> Generator[NDArray, None, None]:
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
        yield from stream_videos(files)
    elif suffix.issubset(image_format):
        yield from stream_images(files)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")
