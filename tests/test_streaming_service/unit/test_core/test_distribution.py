from typing import Any

import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import Component
from cala.streaming.core.distribution import Distributor
from cala.streaming.stores.common import Footprints, Traces
from cala.streaming.stores.odl import (
    PixelStats,
)


class TestDistributor:
    """Test suite for the Distributor class."""

    @pytest.fixture
    def sample_distributor(self) -> Distributor:
        """Create a sample distributor with default parameters."""
        return Distributor()

    @pytest.fixture
    def sample_data(self) -> dict[str, Any]:
        """Create sample data arrays for testing."""
        # Create sample coordinates
        n_components = 3
        height, width = 10, 10
        n_frames = 5

        coords = {
            "id_": ("component", [f"id{i}" for i in range(n_components)]),
            "type_": (
                "component",
                [Component.NEURON.value, Component.NEURON.value, Component.BACKGROUND.value],
            ),
        }

        # Create sample footprints
        footprints_data = np.random.rand(n_components, height, width)
        footprints = xr.DataArray(
            footprints_data, dims=("component", "height", "width"), coords=coords
        )

        # Create sample traces
        traces_data = np.random.rand(n_components, n_frames)
        traces = xr.DataArray(traces_data, dims=("component", "frame"), coords=coords)

        # Create sample pixel stats
        pixel_stats_data = np.random.rand(n_components, height, width)
        pixel_stats = xr.DataArray(
            pixel_stats_data, dims=("component", "height", "width"), coords=coords
        )

        # Create sample component stats
        comp_stats_data = np.random.rand(n_components, n_components)
        component_stats = xr.DataArray(
            comp_stats_data, dims=("component", "component"), coords=coords
        )

        # Create sample residual
        residual_data = np.random.rand(height, width, n_frames)
        residual = xr.DataArray(residual_data, dims=("height", "width", "frame"))

        return {
            "footprints": footprints,
            "traces": traces,
            "pixel_stats": pixel_stats,
            "component_stats": component_stats,
            "residual": residual,
        }

    def test_init_single(
        self, sample_distributor: Distributor, sample_data: dict[str, Any], tmp_path
    ) -> None:
        """Test collecting single DataArray results."""
        # Test collecting each type of Observable
        sample_distributor.init(
            sample_data["footprints"], Footprints, peek_size=100, store_dir=tmp_path
        )
        assert np.array_equal(
            sample_distributor.footprintstore.warehouse, sample_data["footprints"]
        )

        sample_distributor.init(sample_data["traces"], Traces, peek_size=100, store_dir=tmp_path)
        assert np.array_equal(sample_distributor.tracestore.warehouse, sample_data["traces"])

        sample_distributor.init(
            sample_data["pixel_stats"], PixelStats, peek_size=100, store_dir=tmp_path
        )
        assert np.array_equal(
            sample_distributor.pixelstatstore.warehouse, sample_data["pixel_stats"]
        )

    def test_init_multiple(
        self, sample_distributor: Distributor, sample_data: dict[str, Any]
    ) -> None: ...
