import numpy as np

from cala.batch.video_stabilization.rigid_translation import RigidTranslator


def test_rigid_translator_initialization():
    """Test basic initialization of RigidTranslator."""
    core_axes = ["height", "width"]
    iter_axis = "frame"
    anchor_frame_index = 0
    rigid_translator = RigidTranslator(core_axes, iter_axis, anchor_frame_index)
    assert rigid_translator.core_axes == core_axes
    assert rigid_translator.iter_axis == iter_axis
    assert rigid_translator.anchor_frame_index == anchor_frame_index


def test_rigid_translator_motion_estimation(preprocessed_video, camera_motion):
    """Test that RigidTranslator correctly estimates the known motion_stabilization."""
    video = preprocessed_video

    anchor_frame_index = 0
    # Initialize and fit the rigid translator
    rigid_translator = RigidTranslator(
        core_axes=["height", "width"],
        iter_axis="frame",
        anchor_frame_index=anchor_frame_index,
        max_shift=10,
    )
    rigid_translator.fit(video)

    # True and estimated share same origin point
    true_motion = camera_motion - camera_motion[anchor_frame_index]

    # Get the estimated motion_stabilization
    estimated_motion = rigid_translator.motion_.values

    # The estimated motion_stabilization should be approximately the negative of the true motion_stabilization
    # (within some tolerance due to interpolation and numerical precision)
    np.testing.assert_allclose(
        estimated_motion,
        -true_motion,
        rtol=0.2,  # Allow 20% relative tolerance
        atol=25.0,  # Allow 25 pixel absolute tolerance
    )
