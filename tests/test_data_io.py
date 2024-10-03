from pathlib import Path
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numcodecs import blosc

from src.cala.data_io import DataIO, VideoMetadata



@pytest.fixture
def metadata_fixture():
    """Fixture to mock metadata."""
    return VideoMetadata(num_frames=250, height=720, width=1280, channels=3)


@pytest.fixture
def mock_video_proxy():
    """Fixture to mock VideoProxy object."""
    with mock.patch('src.cala.data_io.VideoProxy') as MockVideoProxy:
        mock_video = mock.MagicMock()
        mock_video.n_frames = 100
        mock_video.shape = (100, 480, 640, 1)  # (frames, height, width, channels)
        MockVideoProxy.return_value = mock_video
        yield MockVideoProxy


@pytest.fixture
def data_io_instance(mock_video_proxy):
    """Fixture to create a DataIO instance with mocked video files."""
    video_dir = Path("/fake/path")
    video_files = ["video1.avi", "video2.avi"]
    return DataIO(video_directory=video_dir, video_files=video_files)


def test_data_io_initialization(data_io_instance):
    """Test if DataIO is initialized with correct attributes."""
    assert data_io_instance.video_directory == Path("/fake/path")
    assert data_io_instance.video_files == ["video1.avi", "video2.avi"]
    assert isinstance(data_io_instance.compressor, blosc.Blosc)

def test_metadata(data_io_instance, mock_video_proxy):
    """Test the metadata property of DataIO."""
    # Mock _load_video_metadata method to avoid file access
    with mock.patch.object(DataIO, "_load_video_metadata", return_value=mock_video_proxy.return_value):
        metadata = data_io_instance.metadata

        assert metadata.num_frames == 200  # Since we mock two videos, each with 100 frames
        assert metadata.height == 480
        assert metadata.width == 640
        assert metadata.channels == 1

@patch("src.cala.data_io.VideoProxy")
def test_load_video_metadata(mock_video_proxy, data_io_instance):
    """Test loading video metadata."""
    mock_video = MagicMock()
    mock_video.n_frames = 100
    mock_video.shape = [100, 720, 1280, 3]
    mock_video_proxy.return_value = mock_video

    video_metadata = data_io_instance._load_video_metadata(Path("/fake/path/video1.avi"))

    assert video_metadata.n_frames == 100
    assert video_metadata.shape == [100, 720, 1280, 3]
    mock_video_proxy.assert_called_once_with(path=Path("/fake/path/video1.avi"))


@mock.patch("src.cala.data_io.av.open")
@mock.patch("src.cala.data_io.zarr.open")
@mock.patch.object(DataIO, "_load_video_metadata")
def test_save_data(mock_load_metadata, mock_zarr_open, mock_av_open, data_io_instance):
    """Test the save_data function with mock Zarr and AV."""
    # Mock the _load_video_metadata method to return a mock VideoMetadata object
    mock_video_metadata = mock.MagicMock()
    mock_video_metadata.n_frames = 100  # Simulate 100 frames for each video
    mock_video_metadata.shape = (100, 480, 640, 1)  # Simulate video dimensions (H, W, C)
    mock_load_metadata.return_value = mock_video_metadata

    # Mock container, stream, packet, and frame behavior for av.open
    mock_container = mock.MagicMock()
    mock_stream = mock.MagicMock()
    mock_packet = mock.MagicMock()
    mock_frame = mock.MagicMock()

    # Setup mock frame data to simulate the frame ndarray
    mock_frame.to_ndarray.return_value = np.zeros((480, 640), dtype=np.uint8)  # Frame size: 480x640
    mock_packet.decode.return_value = [mock_frame] * 100  # Simulate 100 frames per packet
    mock_container.demux.return_value = [mock_packet]  # Single packet containing 100 frames
    mock_streams_video = mock.MagicMock()
    mock_streams_video.video = [mock_stream]

    mock_container.streams = mock_streams_video
    mock_av_open.return_value = mock_container

    # Mock Zarr store
    mock_zarr_store = mock.MagicMock()
    mock_zarr_open.return_value = mock_zarr_store

    data_dir = Path("/fake/save")
    file_name = "test_output"

    # Call the save_data method
    data_io_instance.save_data(data_directory=data_dir, data_name=file_name)

    # Assert AV open was called correctly
    mock_av_open.assert_called()

    # Assert Zarr store was opened with correct parameters
    mock_zarr_open.assert_called_with(
        store=f"{data_dir / file_name}.zarr",
        mode="w",
        shape=(200, 480, 640),  # Mocking 200 frames total (2 videos with 100 frames each)
        chunks=(1024, 480, 640),
        dtype=np.uint8,
        compressor=data_io_instance.compressor
    )

    # Check if Zarr store's append method was called for each frame
    assert mock_zarr_store.append.call_count == 200  # Expecting 200 append calls (100 per video)


def test_video_dimension_mismatch(data_io_instance, mock_video_proxy):
    """Test for exception when video dimensions do not match."""
    # Modify the mock to simulate different dimensions in the second video
    mock_video_proxy.return_value.shape = (100, 480, 640, 1)  # First video
    data_io_instance.video_files.append("video3.avi")

    with mock.patch.object(DataIO, "_load_video_metadata", side_effect=[
        mock_video_proxy.return_value,  # First video
        mock.MagicMock(n_frames=100, shape=(100, 500, 640, 1))  # Second video with different height
    ]):
        with pytest.raises(Exception, match="Frame dimensions of .* do not match to its predecessor"):
            _ = data_io_instance.metadata
