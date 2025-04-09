from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray
from PIL import Image


class VideoStream(Protocol):
    """Protocol defining the interface for video streams."""

    def __iter__(self) -> Iterator[NDArray]:
        """Iterate over video frames."""
        ...

    def close(self) -> None:
        """Close the video stream and release resources."""
        ...


class OpenCVStream:
    """OpenCV-based implementation of video streaming."""

    def __init__(self, video_path: Path) -> None:
        """Initialize the video stream.

        Args:
            video_path: Path to the video file

        Raises:
            FileNotFoundError: If video file doesn't exist
            ValueError: If video file cannot be opened
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise ValueError(f"Failed to open video file: {video_path}")

    def __iter__(self) -> Iterator[NDArray]:
        """Iterate over video frames.

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


class TiffStream:
    """Stream implementation for sequence of TIFF files."""

    def __init__(self, tiff_files: list[Path]) -> None:
        """Initialize the TIFF stream.

        Args:
            tiff_files: List of paths to TIFF files

        Raises:
            ValueError: If no TIFF files provided
        """
        if not tiff_files:
            raise ValueError("No TIFF files provided")

        self._files = tiff_files

        # Validate first file to ensure it's readable and get sample shape
        try:
            with Image.open(self._files[0]) as img:
                frame = np.array(img)
                if len(frame.shape) != 2:
                    raise ValueError("TIFF files must be grayscale")
                self._sample_shape = frame.shape
        except Exception as e:
            raise ValueError(f"Failed to read TIFF file: {self._files[0]}") from e

    def __iter__(self) -> Iterator[NDArray]:
        """Iterate over frames in the TIFF sequence.

        Yields:
            NDArray: Next frame from the sequence
        """
        for file_path in self._files:
            with Image.open(file_path) as img:
                frame = np.array(img)
                if len(frame.shape) != 2:
                    raise ValueError(f"File {file_path} is not grayscale")
                if frame.shape != self._sample_shape:
                    raise ValueError(
                        f"Inconsistent frame shape in {file_path}: "
                        f"expected {self._sample_shape}, got {frame.shape}"
                    )
                yield frame

    def close(self) -> None:
        """No resources to close for TIFF sequence."""
        pass


class MultiVideoStream:
    """Handles streaming from multiple video files sequentially."""

    def __init__(self, video_paths: list[Path]) -> None:
        """Initialize multi-video streaming.

        Args:
            video_paths: List of paths to video files

        Raises:
            ValueError: If no video paths provided
        """
        if not video_paths:
            raise ValueError("No video paths provided")

        self._video_paths = video_paths
        self._current_stream: OpenCVStream | None = None

    def __iter__(self) -> Iterator[NDArray]:
        """Iterate over frames from all videos sequentially.

        Yields:
            NDArray: Next frame from the current video

        Raises:
            FileNotFoundError: If any video file doesn't exist
            ValueError: If any video file cannot be opened
        """
        for video_path in self._video_paths:
            self._current_stream = OpenCVStream(video_path)
            yield from self._current_stream
            self._current_stream.close()

    def close(self) -> None:
        """Close the current video stream if open."""
        if self._current_stream is not None:
            self._current_stream.close()


class IO:
    """Main IO class for handling video streaming."""

    def stream(self, files: list[str | Path]) -> VideoStream:
        """Create a video stream from the provided video files.

        Args:
            files: List of file paths

        Returns:
            VideoStream: A stream that yields video frames

        Raises:
            ValueError: If no video files provided
        """
        file_paths = [Path(f) if isinstance(f, str) else f for f in files]

        # Group files by type
        video_files = []
        tiff_files = []

        for path in file_paths:
            suffix = path.suffix.lower()
            if suffix in [".mp4", ".avi", ".mov"]:
                video_files.append(path)
            elif suffix in [".tif", ".tiff"]:
                tiff_files.append(path)
            else:
                raise ValueError(f"Unsupported file type: {suffix}")

        if video_files and not tiff_files:
            if len(video_files) == 1:
                return OpenCVStream(video_files[0])
            return MultiVideoStream(video_files)
        elif tiff_files and not video_files:
            return TiffStream(tiff_files)
        else:
            raise ValueError("Cannot mix video and TIFF files in the same stream")
