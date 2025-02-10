import numpy as np

from cala.batch.video_stabilization.rigid_translation import RigidTranslator


def test_rigid_translator_initialization():
    """Test basic initialization of RigidTranslator."""
    core_axes = ["height", "width"]
    iter_axis = "frames"
    anchor_frame_index = 0
    rigid_translator = RigidTranslator(core_axes, iter_axis, anchor_frame_index)
    assert rigid_translator.core_axes == core_axes
    assert rigid_translator.iter_axis == iter_axis
    assert rigid_translator.anchor_frame_index == anchor_frame_index


def test_rigid_translator_motion_estimation(preprocessed_video):
    """Test that RigidTranslator correctly estimates the known motion."""
    video, ground_truth, metadata = preprocessed_video

    anchor_frame_index = 0
    # Initialize and fit the rigid translator
    rigid_translator = RigidTranslator(
        core_axes=["height", "width"],
        iter_axis="frames",
        anchor_frame_index=anchor_frame_index,
        max_shift=10,
    )
    rigid_translator.fit(video)

    # Get the true motion that was applied
    true_motion = np.column_stack([metadata["motion"]["y"], metadata["motion"]["x"]])
    # True and estimated share same origin point
    true_motion = true_motion - true_motion[anchor_frame_index]

    # Get the estimated motion
    estimated_motion = rigid_translator.motion_.values

    # The estimated motion should be approximately the negative of the true motion
    # (within some tolerance due to interpolation and numerical precision)
    np.testing.assert_allclose(
        estimated_motion,
        -true_motion,
        rtol=0.2,  # Allow 20% relative tolerance
        atol=15.0,  # Allow 15 pixel absolute tolerance
    )


def test_rigid_translator_preserves_neuron_traces(preprocessed_video, stabilized_video):
    """Test that RigidTranslator's correction preserves neuron calcium traces similarly to ground truth."""
    video, ground_truth, _ = preprocessed_video
    ground_truth_stabilized, _, _ = stabilized_video

    # Initialize and fit the rigid translator
    rigid_translator = RigidTranslator(
        core_axes=["height", "width"],
        iter_axis="frames",
        anchor_frame_index=0,
        max_shift=10,
    )
    rigid_translator.fit(video)
    corrected_video = rigid_translator.transform(video)

    # Extract and compare calcium traces from both corrections
    trace_correlations = []

    for n in range(len(ground_truth)):
        y_pos = ground_truth["height"].iloc[n]
        x_pos = ground_truth["width"].iloc[n]
        radius = int(ground_truth["radius"].iloc[n])

        # Function to extract trace from a video
        def extract_trace(vid):
            y_slice = slice(
                max(y_pos - radius, 0), min(y_pos + radius + 1, vid.sizes["height"])
            )
            x_slice = slice(
                max(x_pos - radius, 0), min(x_pos + radius + 1, vid.sizes["width"])
            )
            trace = []
            for f in range(vid.sizes["frames"]):
                region = vid.isel(frames=f)[y_slice, x_slice]
                if not np.any(np.isnan(region)):
                    if len(region) == 0:
                        print("wtf is happening")
                    trace.append(float(region.max()))
                else:
                    trace.append(np.nan)
            return np.array(trace)

        # Extract traces from both corrections
        trace_ours = extract_trace(corrected_video)
        trace_ground_truth = extract_trace(ground_truth_stabilized)

        # Calculate correlation between the traces
        valid_mask = ~np.isnan(trace_ours) & ~np.isnan(trace_ground_truth)
        if np.sum(valid_mask) > 10:  # Only if we have enough valid points
            correlation = np.corrcoef(
                trace_ours[valid_mask], trace_ground_truth[valid_mask]
            )[0, 1]
            trace_correlations.append(correlation)

    # The traces should be highly correlated with ground truth
    assert (
        np.median(trace_correlations) > 0.95
    ), "Calcium traces differ significantly from ground truth stabilization"
