import os

import pytest
import xarray as xr
from sklearn.exceptions import NotFittedError

from cala.streaming.nodes.init.common import (
    FootprintsInitializer,
    FootprintsInitializerParams,
    TracesInitializer,
    TracesInitializerParams,
)


class TestTracesInitializer:
    """Test suite for the TracesInitializer class."""

    @pytest.fixture
    def default_params(self):
        """Create default initialization parameters."""
        return TracesInitializerParams()

    @pytest.fixture
    def default_initializer(self, default_params):
        """Create initializer with default parameters."""
        return TracesInitializer(params=default_params)

    @pytest.fixture
    def footprints_setup(self, stabilized_video):
        """Setup footprints for traces initialization."""
        params = FootprintsInitializerParams()
        initializer = FootprintsInitializer(params=params)
        video = stabilized_video

        initializer.learn_one(frame=video[0])
        footprints = initializer.transform_one()

        return footprints

    def test_initialization(self, default_initializer, default_params):
        """Test basic initialization of TracesInitializer."""
        assert default_initializer.params == default_params
        assert not default_initializer.is_fitted_

    @pytest.mark.viz
    @pytest.mark.parametrize("jit_enabled", [True, False])
    def test_learn_one_basic(
        self,
        default_initializer,
        footprints_setup,
        stabilized_video,
        jit_enabled,
        visualizer,
    ):
        """Test basic learning functionality."""
        if not jit_enabled:
            os.environ["NUMBA_DISABLE_JIT"] = "1"

        video = stabilized_video
        frames = video[0:3]

        default_initializer.learn_one(footprints=footprints_setup, frame=frames)
        traces = default_initializer.transform_one()

        visualizer.plot_traces(traces, subdir="init")
        assert isinstance(traces, xr.DataArray)
        assert (
            traces.sizes[default_initializer.params.component_axis]
            == footprints_setup.sizes[default_initializer.params.component_axis]
        )
        assert traces.sizes[default_initializer.params.frames_axis] == 3

    class TestEdgeCases:
        """Nested test class for edge cases and error conditions."""

        def test_transform_before_learn(self, default_initializer):
            """Test calling transform_one before learn_one."""
            with pytest.raises(NotFittedError):
                default_initializer.transform_one()

        def test_learn_with_mismatched_dimensions(
            self, default_initializer, footprints_setup, stabilized_video
        ):
            """Test learning with mismatched dimensions."""
            video = stabilized_video
            # Modify frames to create dimension mismatch
            frames = video[0:3].drop_isel({"width": [-1]})  # Incorrect shape

            with pytest.raises(ValueError):
                default_initializer.learn_one(footprints=footprints_setup, frame=frames)

    class TestPerformance:
        """Nested test class for performance-related tests."""

        ...
