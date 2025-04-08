import pytest
import xarray as xr

from cala.streaming.nodes.init.common import (
    FootprintsInitializer,
    FootprintsInitializerParams,
)
from cala.viz_util import Visualizer


class TestFootprintsInitializer:
    """Test suite for the FootprintsInitializer class."""

    @pytest.fixture
    def default_params(self) -> FootprintsInitializerParams:
        """Create default initialization parameters."""
        return FootprintsInitializerParams()

    @pytest.fixture
    def custom_params(self) -> FootprintsInitializerParams:
        """Create custom initialization parameters."""
        return FootprintsInitializerParams(
            kernel_size=5, threshold_factor=0.3, distance_mask_size=5
        )

    @pytest.fixture
    def default_initializer(
        self, default_params: FootprintsInitializerParams
    ) -> FootprintsInitializer:
        """Create initializer with default parameters."""
        return FootprintsInitializer(params=default_params)

    @pytest.fixture
    def custom_initializer(
        self, custom_params: FootprintsInitializerParams
    ) -> FootprintsInitializer:
        """Create initializer with custom parameters."""
        return FootprintsInitializer(params=custom_params)

    def test_initialization(
        self,
        default_initializer: FootprintsInitializer,
        default_params: FootprintsInitializerParams,
    ) -> None:
        """Test basic initialization of FootprintsInitializer."""
        assert default_initializer.params == default_params

    def test_learn_one_first_frame(
        self, default_initializer: FootprintsInitializer, stabilized_video: xr.DataArray
    ) -> None:
        """Test learning from the first frame."""
        video = stabilized_video
        first_frame = video[0]

        default_initializer.learn_one(frame=first_frame)

        assert default_initializer.markers_.shape == first_frame.shape
        assert default_initializer.num_markers_ == len(default_initializer.footprints_)

    @pytest.mark.viz
    def test_transform_one_output_shapes(
        self,
        visualizer: Visualizer,
        default_initializer: FootprintsInitializer,
        stabilized_video: xr.DataArray,
    ) -> None:
        """Test output shapes from transform_one."""
        video = stabilized_video
        first_frame = video[0]

        default_initializer.learn_one(frame=first_frame)
        footprints = default_initializer.transform_one()

        visualizer.plot_footprints(footprints, subdir="init")
        # Check shapes match input frame
        assert footprints[0].shape == first_frame.shape

    class TestEdgeCases:
        """Nested test class for edge cases and error conditions."""

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
        def test_invalid_parameters(self, param: dict[str, int]) -> None:
            """Test invalid parameters."""
            with pytest.raises(ValueError):
                FootprintsInitializerParams(**param)

    class TestPerformance:
        """Nested test class for performance-related tests."""

        pass
