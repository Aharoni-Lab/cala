import os

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError

from cala.streaming.initialization import (
    FootprintsInitializer,
    FootprintsInitializerParams,
    TracesInitializer,
    TracesInitializerParams,
)
from cala.streaming.types import Traces
from tests.fixtures import stabilized_video


class TestTracesInitializer:
    """Test suite for the TracesInitializer class."""

    @pytest.fixture
    def default_params(self):
        """Create default initialization parameters."""
        return TracesInitializerParams()

    @pytest.fixture
    def custom_params(self):
        """Create custom initialization parameters."""
        return TracesInitializerParams(
            num_frames_to_use=5,
        )

    @pytest.fixture
    def default_initializer(self, default_params):
        """Create initializer with default parameters."""
        return TracesInitializer(params=default_params)

    @pytest.fixture
    def custom_initializer(self, custom_params):
        """Create initializer with custom parameters."""
        return TracesInitializer(params=custom_params)

    @pytest.fixture
    def footprints_setup(self, stabilized_video):
        """Setup footprints for traces initialization."""
        params = FootprintsInitializerParams()
        initializer = FootprintsInitializer(params=params)
        video, _, _ = stabilized_video

        initializer.learn_one(frame=video[0])
        neuron_footprints, _ = initializer.transform_one()

        return neuron_footprints

    def test_initialization(self, default_initializer, default_params):
        """Test basic initialization of TracesInitializer."""
        assert default_initializer.params == default_params
        assert not default_initializer.is_fitted_

    @pytest.mark.parametrize("jit_enabled", [True, False])
    def test_learn_one_basic(
        self, default_initializer, footprints_setup, stabilized_video, jit_enabled
    ):
        """Test basic learning functionality."""
        if not jit_enabled:
            os.environ["NUMBA_DISABLE_JIT"] = "1"

        video, _, _ = stabilized_video
        frames = video[0:3]

        default_initializer.learn_one(footprints=footprints_setup, frames=frames)
        traces = default_initializer.transform_one()

        assert isinstance(traces, Traces)
        assert (
            traces.sizes[default_initializer.params.component_axis]
            == footprints_setup.sizes[default_initializer.params.component_axis]
        )
        assert traces.sizes[default_initializer.params.frames_axis] == min(
            3, default_initializer.params.num_frames_to_use
        )

    def test_transform_one_output_types(
        self, default_initializer, footprints_setup, stabilized_video
    ):
        """Test output types from transform_one."""
        video, _, _ = stabilized_video
        frames = video[0:3]

        default_initializer.learn_one(footprints=footprints_setup, frames=frames)
        traces = default_initializer.transform_one()

        assert isinstance(traces, Traces)
        assert isinstance(traces.values, np.ndarray)
        assert traces.values.dtype == np.float64

    def test_custom_parameters(
        self, custom_initializer, footprints_setup, stabilized_video
    ):
        """Test initializer with custom parameters."""
        video, _, _ = stabilized_video
        frames = video[0:5]  # Using more frames to test custom num_frames_to_use

        custom_initializer.learn_one(footprints=footprints_setup, frames=frames)
        traces = custom_initializer.transform_one()

        assert (
            traces.sizes[custom_initializer.params.component_axis]
            == footprints_setup.sizes[custom_initializer.params.component_axis]
        )
        assert traces.sizes[custom_initializer.params.frames_axis] == min(
            5, custom_initializer.params.num_frames_to_use
        )

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
            video, _, _ = stabilized_video
            # Modify frames to create dimension mismatch
            frames = video[0:3].drop_isel({"width": [-1]})  # Incorrect shape

            with pytest.raises(ValueError):
                default_initializer.learn_one(
                    footprints=footprints_setup, frames=frames
                )

        @pytest.mark.parametrize(
            "param",
            [{"num_frames_to_use": 0}, {"num_frames_to_use": -1}],
        )
        def test_invalid_parameters(self, param):
            """Test initialization with invalid parameters."""
            with pytest.raises(ValueError):
                TracesInitializerParams(**param)

    class TestPerformance:
        """Nested test class for performance-related tests."""

        ...
