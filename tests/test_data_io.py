import pytest
import tempfile
import shutil
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np

from cala.data_io import DataIO, VideoMetadata


@pytest.fixture
def metadata_fixture():
    """Fixture to mock metadata."""
    return VideoMetadata(num_frames=250, height=720, width=1280, channels=3)


@pytest.fixture
def temp_video_paths():
    # Setup: Create a temporary directory
    temp_dir = tempfile.mkdtemp()
    video_directory = Path(temp_dir)
    video_files = ["video1.mp4", "video2.mp4"]

    # Create empty files to represent video files
    for video_file in video_files:
        video_path = video_directory / video_file
        video_path.touch()

    yield video_directory, video_files

    # Teardown: Remove the temporary directory
    shutil.rmtree(temp_dir)


def test_dataio_initialization():
    video_paths = [Path("/path/to/video1.mp4"), Path("/path/to/video2.mp4")]
    data_io = DataIO(video_paths=video_paths)
    assert data_io.video_paths == video_paths
    assert data_io.compressor is not None


def test_dataio_metadata():
    video_paths = [Path("/path/to/video1.mp4"), Path("/path/to/video2.mp4")]
    data_io = DataIO(video_paths=video_paths)

    # Mock the Image class
    with patch("cala.data_io.Image") as MockImage:
        # Create a mock instance of Image
        mock_image_instance = MagicMock()
        mock_image_instance.n_frames = 10
        mock_image_instance.shape = (10, 1080, 1920, 3)
        MockImage.return_value = mock_image_instance

        metadata = data_io.metadata

        assert metadata.num_frames == 20  # 10 frames from each video
        assert metadata.height == 1080
        assert metadata.width == 1920
        assert metadata.channels == 3


def test_dataio_metadata_inconsistent_dimensions():
    video_paths = [Path("/path/to/video1.mp4"), Path("/path/to/video2.mp4")]
    data_io = DataIO(video_paths=video_paths)

    # Mock the Image class
    with patch("cala.data_io.Image") as MockImage:
        # Create two mock instances of Image with different shapes
        mock_image_instance1 = MagicMock()
        mock_image_instance1.n_frames = 10
        mock_image_instance1.shape = (10, 1080, 1920, 3)

        mock_image_instance2 = MagicMock()
        mock_image_instance2.n_frames = 15
        mock_image_instance2.shape = (15, 720, 1280, 3)

        # Set the return values for the two different video paths
        MockImage.side_effect = [mock_image_instance1, mock_image_instance2]

        with pytest.raises(Exception) as exc_info:
            _ = data_io.metadata

        assert "Frame dimensions of" in str(exc_info.value)


def test_dataio_save_data(tmp_path):
    video_paths = [Path("/path/to/video1.mp4")]
    data_io = DataIO(video_paths=video_paths)

    # Mock the Image class and av.open
    with patch("cala.data_io.Image") as MockImage, patch(
        "cala.data_io.av.open"
    ) as mock_av_open, patch("cala.data_io.tqdm", lambda x, *args, **kwargs: x):

        # Mock Image
        mock_image_instance = MagicMock()
        mock_image_instance.n_frames = 10
        mock_image_instance.shape = (10, 1080, 1920, 3)
        MockImage.return_value = mock_image_instance

        # Mock av.open and related methods
        mock_container = MagicMock()
        mock_video_stream = MagicMock()
        mock_container.streams.video = [mock_video_stream]
        mock_av_open.return_value = mock_container

        # Mock demuxed packets and frames
        mock_frame = MagicMock()
        mock_frame.to_ndarray.return_value = np.zeros((1080, 1920), dtype=np.uint8)
        mock_packet = MagicMock()
        mock_packet.decode.return_value = [mock_frame]
        mock_container.demux.return_value = [mock_packet] * 10  # Simulate 10 packets

        # Now call save_data
        data_directory = tmp_path
        data_name = "test_video"
        data_io.save_data(data_directory=data_directory, data_name=data_name)

        # Check that the zarr store was created
        zarr_file = data_directory / f"{data_name}.zarr"
        assert zarr_file.exists()

        # Open the zarr store and check the shape
        import zarr

        zarr_store = zarr.open(store=str(zarr_file), mode="r")
        assert zarr_store.shape == (10, 1080, 1920)
        # Optionally, check the data
        data = zarr_store[:]
        assert data.shape == (10, 1080, 1920)
        assert (data == 0).all()
