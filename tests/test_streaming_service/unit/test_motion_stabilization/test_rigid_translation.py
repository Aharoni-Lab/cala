import numpy as np
import pytest
import xarray as xr

from cala.streaming.motion_stabilization import RigidTranslatorParams, RigidTranslator


class TestMotionStabilizer:
    @pytest.fixture
    def default_params(self):
        """Create default parameters for testing"""
        params = RigidTranslatorParams(max_shift=10)
        return params

    @pytest.fixture
    def default_stabilizer(self, default_params):
        """Create BackgroundEraser instance with uniform method"""
        return RigidTranslator(default_params)

    def test_initialization(self, default_params):
        """Test proper initialization of BackgroundEraser"""
        stabilizer = RigidTranslator(default_params)
        assert stabilizer.params.max_shift == default_params.max_shift
        assert stabilizer._learn_count == 0
        assert stabilizer._transform_count == 0
        assert stabilizer.motion_ == []

    def test_rigid_translator_motion_estimation(
        self, default_stabilizer, preprocessed_video
    ):
        """Test that RigidTranslator correctly estimates the known motion_stabilization."""
        video, ground_truth, metadata = preprocessed_video

        transformed_frames = []
        # Initialize and fit the rigid translator
        for frame in video:
            transformed_frames.append(
                default_stabilizer.learn_one(frame).transform_one(frame)
            )

        # Get the true motion_stabilization that was applied
        true_motion = np.column_stack(
            [
                metadata["motion"]["y"],
                metadata["motion"]["x"],
            ]
        )
        # True and estimated share same origin point
        true_motion = true_motion - true_motion[0]

        # Get the estimated motion_stabilization
        estimated_motion = default_stabilizer.motion_

        # The estimated motion_stabilization should be approximately the negative of the true motion_stabilization
        # (within some tolerance due to interpolation and numerical precision)
        np.testing.assert_allclose(
            estimated_motion,
            -true_motion[1:],  # skip the first frame
            rtol=0.2,  # Allow 20% relative tolerance
            atol=15.0,  # Allow 15 pixel absolute tolerance
        )

    def test_rigid_translator_preserves_neuron_traces(
        self, default_stabilizer, preprocessed_video, stabilized_video
    ):
        """Test that RigidTranslator's correction preserves neuron calcium traces similarly to ground truth."""
        video, ground_truth, _ = preprocessed_video
        ground_truth_stabilized, _, _ = stabilized_video

        corrected_video = []
        for frame in video:
            corrected_video.append(
                default_stabilizer.learn_one(frame).transform_one(frame)
            )

        corrected_video = xr.DataArray(
            corrected_video, dims=video.dims, coords=video.coords
        )

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
