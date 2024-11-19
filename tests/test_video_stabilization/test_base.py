import pytest
import numpy as np
import xarray as xr
from cala.video_stabilization.base import BaseMotionCorrector


# Create a mock subclass for testing purposes
class MockMotionCorrector(BaseMotionCorrector):
    def __init__(self, core_axes, iter_axis, anchor_frame_index):
        super().__init__(core_axes, iter_axis, anchor_frame_index)

    def _fit_kernel(self, current_frame: np.ndarray, **kwargs) -> np.ndarray:
        # Mock fit: return zeros
        shift_size = 2
        return np.zeros(shift_size)

    def _transform_kernel(self, frame: np.ndarray, shift: np.ndarray) -> np.ndarray:
        # Mock transform: return frame unchanged
        return frame


def create_synthetic_data(
    num_frames=5, height=10, width=10, iter_axis="frames", core_axes=["height", "width"]
):
    data = np.random.rand(num_frames, height, width)
    coords = {
        iter_axis: np.arange(num_frames),
        core_axes[0]: np.arange(height),
        core_axes[1]: np.arange(width),
    }
    return xr.DataArray(data, dims=[iter_axis] + core_axes, coords=coords)


def test_base_motion_corrector_initialization():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 2
    mock_corrector = MockMotionCorrector(core_axes, iter_axis, anchor_frame_index)
    assert mock_corrector.core_axes == core_axes
    assert mock_corrector.iter_axis == iter_axis
    assert mock_corrector.anchor_frame_index == anchor_frame_index
    assert mock_corrector.anchor_frame_ is None
    assert mock_corrector.motion_ is None


def test_base_motion_corrector_fit():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 1
    mock_corrector = MockMotionCorrector(core_axes, iter_axis, anchor_frame_index)
    X = create_synthetic_data(
        num_frames=3, height=5, width=5, iter_axis=iter_axis, core_axes=core_axes
    )
    mock_corrector.fit(X)
    assert mock_corrector.anchor_frame_ is not None
    assert mock_corrector.motion_ is not None
    # Check that motion_ has the correct shape
    assert "shift_dim" in mock_corrector.motion_.dims
    assert mock_corrector.motion_.sizes[iter_axis] == X.sizes[iter_axis]


def test_base_motion_corrector_transform_not_fitted():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 0
    mock_corrector = MockMotionCorrector(core_axes, iter_axis, anchor_frame_index)
    X = create_synthetic_data(
        num_frames=3, height=5, width=5, iter_axis=iter_axis, core_axes=core_axes
    )
    with pytest.raises(
        ValueError,
        match="Motion has not been calculated yet. Fit method must be run before transform.",
    ):
        mock_corrector.transform(X)


def test_base_motion_corrector_fit_transform():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 0
    mock_corrector = MockMotionCorrector(core_axes, iter_axis, anchor_frame_index)
    X = create_synthetic_data(
        num_frames=3, height=5, width=5, iter_axis=iter_axis, core_axes=core_axes
    )
    mock_corrector.fit(X)
    transformed_X = mock_corrector.transform(X)
    # Since _transform_kernel is a mock that returns the frame unchanged, transformed_X should be equal to X
    xr.testing.assert_equal(transformed_X, X)


def test_anchor_by_index_invalid_index():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 10  # Out of bounds
    mock_corrector = MockMotionCorrector(core_axes, iter_axis, anchor_frame_index)
    X = create_synthetic_data(
        num_frames=5, height=5, width=5, iter_axis=iter_axis, core_axes=core_axes
    )
    with pytest.raises(IndexError, match="anchor_index is out of bounds."):
        mock_corrector.anchor_by_index(X, anchor_index=anchor_frame_index)
