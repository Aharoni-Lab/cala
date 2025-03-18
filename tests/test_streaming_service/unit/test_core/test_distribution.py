import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import Distributor, Footprints, Traces
from cala.streaming.stores.odl import PixelStats, ComponentStats, Residual


class TestDataExchangeInitialization:
    def test_default_initialization(self):
        exchange = Distributor()
        assert exchange.component_axis == "components"
        assert exchange.spatial_axes == ("width", "height")
        assert exchange.frame_axis == "frames"
        assert isinstance(exchange.footprints, Footprints)
        assert isinstance(exchange.traces, Traces)

    def test_custom_axes_initialization(self):
        exchange = Distributor(
            component_axis="cells", spatial_axes=("x", "y"), frame_axis="time"
        )
        assert exchange.component_axis == "cells"
        assert exchange.spatial_axes == ("x", "y")
        assert exchange.frame_axis == "time"


class TestDistributor:
    """Test suite for the Distributor class."""

    @pytest.fixture
    def sample_distributor(self):
        """Create a sample distributor with default parameters."""
        return Distributor()

    @pytest.fixture
    def sample_data(self):
        """Create sample data arrays for testing."""
        # Create sample coordinates
        n_components = 3
        height, width = 10, 10
        n_frames = 5

        coords = {
            "id_": ("components", [f"id{i}" for i in range(n_components)]),
            "type_": ("components", ["neuron", "neuron", "background"]),
        }

        # Create sample footprints
        footprints_data = np.random.rand(n_components, height, width)
        footprints = Footprints(
            footprints_data, dims=("components", "height", "width"), coords=coords
        )

        # Create sample traces
        traces_data = np.random.rand(n_components, n_frames)
        traces = Traces(traces_data, dims=("components", "frames"), coords=coords)

        # Create sample pixel stats
        pixel_stats_data = np.random.rand(n_components, height, width)
        pixel_stats = PixelStats(
            pixel_stats_data, dims=("components", "height", "width"), coords=coords
        )

        # Create sample component stats
        comp_stats_data = np.random.rand(n_components, n_components)
        component_stats = ComponentStats(
            comp_stats_data, dims=("components", "components"), coords=coords
        )

        # Create sample residual
        residual_data = np.random.rand(height, width, n_frames)
        residual = Residual(residual_data, dims=("height", "width", "frames"))

        return {
            "footprints": footprints,
            "traces": traces,
            "pixel_stats": pixel_stats,
            "component_stats": component_stats,
            "residual": residual,
        }

    def test_initialization(self, sample_distributor):
        """Test proper initialization of Distributor."""
        assert sample_distributor.component_axis == "components"
        assert sample_distributor.spatial_axes == ("width", "height")
        assert sample_distributor.frame_axis == "frames"
        assert sample_distributor.id_coord == "id_"
        assert sample_distributor.type_coord == "type_"

        # Test store initialization
        assert isinstance(sample_distributor.footprints, Footprints)
        assert isinstance(sample_distributor.traces, Traces)
        assert isinstance(sample_distributor.pixel_stats, PixelStats)
        assert isinstance(sample_distributor.component_stats, ComponentStats)
        assert isinstance(sample_distributor.residual, Residual)

    def test_get_observable(self, sample_distributor, sample_data):
        """Test retrieving Observable instances by type."""
        # Test getting each type of Observable
        assert (
                (sample_distributor.get(Footprints) == sample_distributor.footprints)
                | (
                        np.isnan(sample_distributor.get(Footprints))
                        & np.isnan(sample_distributor.footprints)
                )
        ).all()
        assert (
                (sample_distributor.get(Traces) == sample_distributor.traces)
                | (
                        np.isnan(sample_distributor.get(Traces))
                        & np.isnan(sample_distributor.traces)
                )
        ).all()
        assert (
                (sample_distributor.get(PixelStats) == sample_distributor.pixel_stats)
                | (
                        np.isnan(sample_distributor.get(PixelStats))
                        & np.isnan(sample_distributor.pixel_stats)
                )
        ).all()
        assert (
                (
                        sample_distributor.get(ComponentStats)
                        == sample_distributor.component_stats
                )
                | (
                        np.isnan(sample_distributor.get(ComponentStats))
                        & np.isnan(sample_distributor.component_stats)
            )
        ).all()
        assert (
                (sample_distributor.get(Residual) == sample_distributor.residual)
                | (
                        np.isnan(sample_distributor.get(Residual))
                        & np.isnan(sample_distributor.residual)
                )
        ).all()

        # Test getting non-existent type
        class DummyObservable(xr.DataArray):
            pass

        assert sample_distributor.get(DummyObservable) is None

    def test_collect_single(self, sample_distributor, sample_data):
        """Test collecting single DataArray results."""
        # Test collecting each type of Observable
        sample_distributor.collect(sample_data["footprints"])
        assert np.array_equal(sample_distributor.footprints, sample_data["footprints"])

        sample_distributor.collect(sample_data["traces"])
        assert np.array_equal(sample_distributor.traces, sample_data["traces"])

        sample_distributor.collect(sample_data["pixel_stats"])
        assert np.array_equal(
            sample_distributor.pixel_stats, sample_data["pixel_stats"]
        )

    def test_collect_multiple(self, sample_distributor, sample_data):
        """Test collecting multiple DataArray results at once."""
        # Test collecting multiple Observables
        sample_distributor.collect(
            (
                sample_data["footprints"],
                sample_data["traces"],
                sample_data["pixel_stats"],
            )
        )

        assert np.array_equal(sample_distributor.footprints, sample_data["footprints"])
        assert np.array_equal(sample_distributor.traces, sample_data["traces"])
        assert np.array_equal(
            sample_distributor.pixel_stats, sample_data["pixel_stats"]
        )

    def test_collect_invalid(self, sample_distributor):
        """Test collecting invalid data types."""
        # Test with invalid data type
        invalid_data = xr.DataArray(np.random.rand(5, 5))
        sample_distributor.collect(
            invalid_data
        )  # Should not raise error but not store anything

        # Test with invalid tuple
        invalid_tuple = (np.random.rand(5, 5), "invalid")
        sample_distributor.collect(
            invalid_tuple
        )  # Should not raise error but not store anything

    def test_coordinate_consistency(self, sample_distributor, sample_data):
        """Test that collected data maintains coordinate consistency."""
        sample_distributor.collect(sample_data["footprints"])
        sample_distributor.collect(sample_data["traces"])

        # Check that coordinates are preserved
        assert np.array_equal(
            sample_distributor.footprints.coords[sample_distributor.id_coord],
            sample_data["footprints"].coords[sample_distributor.id_coord],
        )
        assert np.array_equal(
            sample_distributor.traces.coords[sample_distributor.type_coord],
            sample_data["traces"].coords[sample_distributor.type_coord],
        )
