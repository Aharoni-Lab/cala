import pytest
import numpy as np
import xarray as xr
import cv2
from cala.motion_correction.rigid_translation import RigidTranslator


def create_shifted_frame(anchor_frame, shift_y, shift_x):
    M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    shifted_frame = cv2.warpAffine(
        anchor_frame,
        M,
        (anchor_frame.shape[1], anchor_frame.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return shifted_frame


def create_synthetic_video(
    num_frames=3,
    height=20,
    width=20,
    iter_axis="frames",
    core_axes=("height", "width"),
    shifts=None,
):
    anchor_frame = np.random.rand(height, width)
    frames = [anchor_frame]
    if shifts is None:
        shifts = [(0, 0), (1, -1), (-2, 2)]  # Example shifts
    for shift in shifts[1:]:
        shifted_frame = create_shifted_frame(
            anchor_frame, shift_y=shift[0], shift_x=shift[1]
        )
        frames.append(shifted_frame)
    data = np.stack(frames, axis=0)
    coords = {
        iter_axis: np.arange(num_frames),
        core_axes[0]: np.arange(height),
        core_axes[1]: np.arange(width),
    }
    return xr.DataArray(data, dims=(iter_axis, *core_axes), coords=coords)


def test_rigid_translator_initialization():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 0
    rigid_translator = RigidTranslator(core_axes, iter_axis, anchor_frame_index)
    assert rigid_translator.core_axes == core_axes
    assert rigid_translator.iter_axis == iter_axis
    assert rigid_translator.anchor_frame_index == anchor_frame_index
    assert rigid_translator.anchor_frame_ is None
    assert rigid_translator.motion_ is None


def test_rigid_translator_fit():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 0
    rigid_translator = RigidTranslator(core_axes, iter_axis, anchor_frame_index)
    shifts = [(0.0, 0.0), (2.0, -3.0), (-1.0, 4.0)]
    X = create_synthetic_video(
        num_frames=3,
        height=10,
        width=10,
        iter_axis=iter_axis,
        core_axes=core_axes,
        shifts=shifts,
    )
    rigid_translator.fit(X)
    assert rigid_translator.anchor_frame_ is not None
    assert rigid_translator.motion_ is not None
    # Check motion_ dimensions
    assert "shift_dim" in rigid_translator.motion_.dims
    assert rigid_translator.motion_.sizes[iter_axis] == X.sizes[iter_axis]
    # Check that motion_ values match the shifts
    expected_shifts = -1 * np.array(shifts)  # multiplying negative one to reverse.
    np.testing.assert_array_almost_equal(
        rigid_translator.motion_.values, expected_shifts, decimal=1
    )


def test_rigid_translator_transform():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 0
    rigid_translator = RigidTranslator(core_axes, iter_axis, anchor_frame_index)
    shifts = [(0, 0), (2, -3), (-1, 4)]
    X = create_synthetic_video(
        num_frames=3,
        height=10,
        width=10,
        iter_axis=iter_axis,
        core_axes=core_axes,
        shifts=shifts,
    )
    rigid_translator.fit(X)
    transformed_X = rigid_translator.transform(X)
    # Since the shifts are known, applying the inverse shift should recover the base frame
    # Here, since RigidTranslator applies shifts to align frames to the base, transformed_X should be similar to the anchor_frame
    anchor_frame = rigid_translator.anchor_frame_.values
    for i in range(X.sizes[iter_axis]):
        frame = transformed_X.isel({iter_axis: i}).values
        mask = ~np.isnan(frame) & ~np.isnan(anchor_frame)
        frame_no_nan = frame[mask]
        anchor_frame_no_nan = anchor_frame[mask]
        np.testing.assert_array_almost_equal(
            frame_no_nan, anchor_frame_no_nan, decimal=1
        )


def test_rigid_translator_fit_transform():
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 0
    rigid_translator = RigidTranslator(core_axes, iter_axis, anchor_frame_index)
    shifts = [(0, 0), (3, -2), (-2, 5)]
    X = create_synthetic_video(
        num_frames=3,
        height=15,
        width=15,
        iter_axis=iter_axis,
        core_axes=core_axes,
        shifts=shifts,
    )
    rigid_translator.fit(X)
    transformed_X = rigid_translator.transform(X)
    # Verify that all transformed frames align with the base frame
    anchor_frame = rigid_translator.anchor_frame_.values
    for i in range(X.sizes[iter_axis]):
        frame = transformed_X.isel({iter_axis: i}).values
        # Allow some tolerance due to interpolation
        difference = np.abs(frame - anchor_frame)
        mask = ~np.isnan(difference)
        assert np.all(difference[mask] < 0.1), f"Frame {i} is not properly aligned."
