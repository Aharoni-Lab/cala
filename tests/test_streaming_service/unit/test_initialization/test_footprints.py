import pytest
from sklearn.exceptions import NotFittedError

from cala.streaming.initialization import (
    FootprintsInitializer,
    FootprintsInitializerParams,
)
from cala.streaming.types import NeuronFootprints, BackgroundFootprints
from tests.fixtures import stabilized_video


class TestFootprintsInitializer:
    """Test suite for the FootprintsInitializer class."""

    @pytest.fixture
    def default_params(self):
        """Create default initialization parameters."""
        return FootprintsInitializerParams()

    @pytest.fixture
    def custom_params(self):
        """Create custom initialization parameters."""
        return FootprintsInitializerParams(
            kernel_size=5, threshold_factor=0.3, distance_mask_size=5
        )

    @pytest.fixture
    def default_initializer(self, default_params):
        """Create initializer with default parameters."""
        return FootprintsInitializer(params=default_params)

    @pytest.fixture
    def custom_initializer(self, custom_params):
        """Create initializer with custom parameters."""
        return FootprintsInitializer(params=custom_params)

    def test_initialization(self, default_initializer, default_params):
        """Test basic initialization of FootprintsInitializer."""
        assert default_initializer.params == default_params

    def test_learn_one_first_frame(self, default_initializer, stabilized_video):
        """Test learning from the first frame."""
        video, _, _ = stabilized_video
        first_frame = video[0]

        default_initializer.learn_one(frame=first_frame)

        assert default_initializer.markers_.shape == first_frame.shape
        assert default_initializer.num_markers_ == len(
            default_initializer.neurons_
        ) + len(default_initializer.background_)

    def test_transform_one_output_shapes(self, default_initializer, stabilized_video):
        """Test output shapes from transform_one."""
        video, _, _ = stabilized_video
        first_frame = video[0]

        default_initializer.learn_one(frame=first_frame)
        neuron_footprints, background_footprints = default_initializer.transform_one()

        # Check shapes match input frame
        assert neuron_footprints[0].shape == first_frame.shape
        assert background_footprints[0].shape == first_frame.shape

    def test_transform_one_output_types(self, default_initializer, stabilized_video):
        """Test output types from transform_one."""
        video, _, _ = stabilized_video
        first_frame = video[0]

        default_initializer.learn_one(frame=first_frame)
        neuron_footprints, background_footprints = default_initializer.transform_one()

        # Check types
        assert isinstance(neuron_footprints, NeuronFootprints)
        assert isinstance(background_footprints, BackgroundFootprints)

    class TestEdgeCases:
        """Nested test class for edge cases and error conditions."""

        def test_transform_before_learn(self, default_initializer):
            """Test calling transform_one before learn_one."""
            with pytest.raises(NotFittedError):
                default_initializer.transform_one()

        @pytest.mark.parametrize(
            "param",
            [
                {"kernel_size": 0},
                {"kernel_size": -1},
                {"threshold_factor": 0},
                {"threshold_factor": -1},
                {"distance_mask_size": 0},
                {"distance_mask_size": -1},
            ],
        )
        def test_invalid_paramters(self, param):
            """Test invalid parameters."""
            with pytest.raises(ValueError):
                FootprintsInitializerParams(**param)

    class TestPerformance:
        """Nested test class for performance-related tests."""

        pass
