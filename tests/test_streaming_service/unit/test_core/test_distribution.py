import numpy as np
import pytest
import xarray as xr

from cala.streaming.core import FootprintStore, TraceStore, Footprints, Traces
from cala.streaming.core.distribution import Distributor
from cala.streaming.stores.odl import (
    PixelStatStore,
    ComponentStatStore,
    ResidualStore,
    PixelStats,
    ComponentStats,
    Residuals,
)


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
        footprints = FootprintStore(
            footprints_data, dims=("components", "height", "width"), coords=coords
        )

        # Create sample traces
        traces_data = np.random.rand(n_components, n_frames)
        traces = TraceStore(traces_data, dims=("components", "frames"), coords=coords)

        # Create sample pixel stats
        pixel_stats_data = np.random.rand(n_components, height, width)
        pixel_stats = PixelStatStore(
            pixel_stats_data, dims=("components", "height", "width"), coords=coords
        )

        # Create sample component stats
        comp_stats_data = np.random.rand(n_components, n_components)
        component_stats = ComponentStatStore(
            comp_stats_data, dims=("components", "components"), coords=coords
        )

        # Create sample residual
        residual_data = np.random.rand(height, width, n_frames)
        residual = ResidualStore(residual_data, dims=("height", "width", "frames"))

        return {
            "footprints": footprints,
            "traces": traces,
            "pixel_stats": pixel_stats,
            "component_stats": component_stats,
            "residual": residual,
        }

    def test_get_observable(self, sample_distributor, sample_data):
        """Test retrieving Observable instances by type."""
        # Test getting each type of Observable
        assert (
            (sample_distributor.get(Footprints) == sample_distributor.footprintstore)
            | (
                np.isnan(sample_distributor.get(Footprints))
                & np.isnan(sample_distributor.footprintstore)
            )
        ).all()
        assert (
            (sample_distributor.get(Traces) == sample_distributor.tracestore)
            | (
                np.isnan(sample_distributor.get(Traces))
                & np.isnan(sample_distributor.tracestore)
            )
        ).all()
        assert (
            (sample_distributor.get(PixelStats) == sample_distributor.pixelstatstore)
            | (
                np.isnan(sample_distributor.get(PixelStats))
                & np.isnan(sample_distributor.pixelstatstore)
            )
        ).all()
        assert (
            (
                sample_distributor.get(ComponentStats)
                == sample_distributor.componentstatstore
            )
            | (
                np.isnan(sample_distributor.get(ComponentStats))
                & np.isnan(sample_distributor.componentstatstore)
            )
        ).all()
        assert (
            (sample_distributor.get(Residuals) == sample_distributor.residualstore)
            | (
                np.isnan(sample_distributor.get(Residuals))
                & np.isnan(sample_distributor.residualstore)
            )
        ).all()

        # Test getting non-existent type
        class DummyObservable(xr.DataArray):
            pass

        assert sample_distributor.get(DummyObservable) is None

    def test_collect_single(self, sample_distributor, sample_data):
        """Test collecting single DataArray results."""
        # Test collecting each type of Observable
        sample_distributor.init(sample_data["footprints"])
        assert np.array_equal(sample_distributor.footprints, sample_data["footprints"])

        sample_distributor.init(sample_data["traces"])
        assert np.array_equal(sample_distributor.traces, sample_data["traces"])

        sample_distributor.init(sample_data["pixel_stats"])
        assert np.array_equal(
            sample_distributor.pixel_stats, sample_data["pixel_stats"]
        )

    def test_collect_multiple(self, sample_distributor, sample_data):
        """Test collecting multiple DataArray results at once."""
        # Test collecting multiple Observables
        sample_distributor.init(
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
        sample_distributor.init(
            invalid_data
        )  # Should not raise error but not store anything

        # Test with invalid tuple
        invalid_tuple = (np.random.rand(5, 5), "invalid")
        sample_distributor.init(
            invalid_tuple
        )  # Should not raise error but not store anything

    def test_coordinate_consistency(self, sample_distributor, sample_data):
        """Test that collected data maintains coordinate consistency."""
        sample_distributor.init(sample_data["footprints"])
        sample_distributor.init(sample_data["traces"])

        # Check that coordinates are preserved
        assert np.array_equal(
            sample_distributor.footprints.coords[sample_distributor.id_coord],
            sample_data["footprints"].coords[sample_distributor.id_coord],
        )
        assert np.array_equal(
            sample_distributor.traces.coords[sample_distributor.type_coord],
            sample_data["traces"].coords[sample_distributor.type_coord],
        )
