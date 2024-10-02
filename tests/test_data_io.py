from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from numcodecs import blosc
from pytest_lazyfixture import lazy_fixture

from src.data_io import DataIO, VideoMetadata


@pytest.fixture
def dataio_instance():
    """Fixture to create a DataIO instance."""
    video_directory = Path("/fake/path")
    video_files = ["video1.avi", "video2.avi"]
    return DataIO(video_directory=video_directory, video_files=video_files)


def test_dataio_initialization(dataio_instance):
    """Test if DataIO is initialized with correct attributes."""
    assert dataio_instance.video_directory == Path("/fake/path")
    assert dataio_instance.video_files == ["video1.avi", "video2.avi"]
    assert isinstance(dataio_instance.compressor, blosc.Blosc)


@patch("src.data_io.VideoProxy")
def test_load_video_metadata(mock_video_proxy, dataio_instance):
    """Test loading video metadata."""
    mock_video = MagicMock()
    mock_video.n_frames = 100
    mock_video.shape = [100, 720, 1280, 3]
    mock_video_proxy.return_value = mock_video

    video_metadata = dataio_instance._load_video_metadata(Path("/fake/path/video1.avi"))

    assert video_metadata.n_frames == 100
    assert video_metadata.shape == [100, 720, 1280, 3]
    mock_video_proxy.assert_called_once_with(path=Path("/fake/path/video1.avi"))


@patch("src.data_io.VideoProxy")
def test_metadata_property(mock_video_proxy, dataio_instance):
    """Test the metadata property that aggregates video metadata."""
    mock_video1 = MagicMock()
    mock_video1.n_frames = 100
    mock_video1.shape = [100, 720, 1280, 3]

    mock_video2 = MagicMock()
    mock_video2.n_frames = 150
    mock_video2.shape = [150, 720, 1280, 3]

    mock_video_proxy.side_effect = [mock_video1, mock_video2]

    metadata = dataio_instance.metadata

    assert metadata.num_frames == 250
    assert metadata.height == 720
    assert metadata.width == 1280
    assert metadata.channels == 3


@patch("src.data_io.zarr.open")
@patch("src.data_io.av.open")
@patch("src.data_io.DataIO.metadata", new_callable=lazy_fixture("metadata_fixture"))
def test_save_data(
    mock_metadata, mock_av_open, mock_zarr_open, dataio_instance, tmp_path
):
    """Test the save_data function."""
    mock_container = MagicMock()
    mock_stream = MagicMock()
    mock_packet = MagicMock()
    mock_frame = MagicMock()

    # Mock the return values for video stream and packets
    mock_frame.to_ndarray.return_value = np.ones((720, 1280), dtype=np.uint8)
    mock_packet.decode.return_value = [mock_frame]
    mock_container.demux.return_value = [mock_packet]
    mock_av_open.return_value = mock_container
    mock_stream.video = [mock_stream]

    mock_zarr_store = MagicMock()
    mock_zarr_open.return_value = mock_zarr_store

    # Call the method being tested
    dataio_instance.save_data(save_directory=tmp_path, save_name="test_save")

    # Check if zarr.open and av.open were called correctly
    mock_zarr_open.assert_called_once_with(
        store=f"{tmp_path / 'test_save'}.zarr",
        mode="w",
        shape=(mock_metadata.num_frames, 720, 1280),
        chunks=(1024, 720, 1280),
        dtype=np.uint8,
        compressor=dataio_instance.compressor,
    )
    mock_av_open.assert_called()

    # Ensure append was called with correct frame data
    assert mock_zarr_store.append.call_count == 2
    mock_container.close.assert_called_once()


@pytest.fixture
def metadata_fixture():
    """Fixture to mock metadata."""
    return VideoMetadata(num_frames=250, height=720, width=1280, channels=3)
