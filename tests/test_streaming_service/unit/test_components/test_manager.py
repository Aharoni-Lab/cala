import numpy as np
import pytest
import xarray as xr

from cala.streaming.core.components.categories import ComponentType, Neuron, Background
from cala.streaming.core.components.manager import StoreManager
from cala.streaming.core.components.observables import FootprintStore, TraceStore
from cala.streaming.core.components.registry import Registry


class TestComponentManager:
    @pytest.fixture
    def empty_manager(self):
        """Create an empty ComponentManager."""
        return StoreManager()

    @pytest.fixture
    def sample_footprint(self):
        """Create a sample footprint for testing."""
        data = np.zeros((10, 10))
        data[4:7, 4:7] = 1  # 3x3 square of ones
        return xr.DataArray(
            data,
            dims=["height", "width"],
            coords={"height": range(10), "width": range(10)},
        )

    @pytest.fixture
    def sample_trace(self):
        """Create a sample trace for testing."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        return xr.DataArray(
            data,
            dims=["frames"],
            coords={"frames": range(len(data))},
        )

    @pytest.fixture
    def populated_manager(self, empty_manager, sample_footprint, sample_trace):
        """Create a ComponentManager with multiple components."""
        # Create components with distinct patterns
        components = []
        for i in range(3):
            # Create footprint with unique pattern
            footprint = sample_footprint.copy()
            footprint[i : i + 3, i : i + 3] = i + 1

            # Create time trace with unique pattern
            trace = sample_trace.copy()
            trace[:] = i + 1

            # Alternate between Neuron and Background
            component = (
                Neuron(detected_frame_idx=i)
                if i % 2 == 0
                else Background(detected_frame_idx=i)
            )
            empty_manager.add_component(component, footprint, trace)
            components.append(component)

        return empty_manager, components

    def test_initialization(self, empty_manager):
        """Test initialization of ComponentManager."""
        assert empty_manager.component_axis == "components"
        assert empty_manager.spatial_axes == ("width", "height")
        assert empty_manager.frame_axis == "frames"
        assert empty_manager.n_components == 0
        assert empty_manager.component_ids == set()
        assert isinstance(empty_manager._registry, Registry)
        assert isinstance(empty_manager._footprints, FootprintStore)
        assert isinstance(empty_manager._traces, TraceStore)

    def test_footprints_dimensions(self, empty_manager):
        """Test footprints dimensions property."""
        assert empty_manager.footprints_dimensions == ("components", "width", "height")

    def test_traces_dimensions(self, empty_manager):
        """Test traces dimensions property."""
        assert empty_manager.traces_dimensions == ("components", "frames")

    def test_verify_component_consistency(self, populated_manager):
        """Test component consistency verification."""
        manager, _ = populated_manager

        # Should not raise error when consistent
        manager.verify_component_consistency()

        # Test inconsistent footprints and traces
        footprint_data = np.zeros((1, 10, 10))
        trace_data = np.zeros((2, 5))  # Different number of components
        manager._footprints.initialize(
            xr.DataArray(
                footprint_data,
                dims=["components", "height", "width"],
                coords={"components": [1], "height": range(10), "width": range(10)},
            )
        )
        manager._traces.initialize([1, 2])  # Different component IDs

        with pytest.raises(
            ValueError, match="Component IDs in footprints and traces must match"
        ):
            manager.verify_component_consistency()

    def test_populate_from_footprints(self, empty_manager, sample_footprint):
        """Test populating from footprints."""
        # Create test footprints
        footprints = sample_footprint.expand_dims(
            dim={"components": [1, 2]},
        )

        # Test initial population
        empty_manager.populate_from_footprints(footprints, ComponentType.NEURON)
        neuron_ids = list(empty_manager.component_ids)
        assert empty_manager.n_components == 2
        assert all(
            isinstance(empty_manager.get_component(i), Neuron) for i in neuron_ids
        )

        # Test updating existing and adding new
        updated_footprints = sample_footprint.expand_dims(
            dim={"components": [3]},  # add 3
        )
        empty_manager.populate_from_footprints(
            updated_footprints, ComponentType.BACKGROUND
        )
        assert empty_manager.n_components == 3
        assert isinstance(
            empty_manager.get_component(neuron_ids[1]), Neuron
        )  # Should not change type
        assert len(empty_manager.background_ids) == 1

    def test_populate_from_traces(self, populated_manager):
        """Test populating traces."""
        manager, _ = populated_manager

        # Create test traces
        new_traces = xr.DataArray(
            np.ones((3, 2)),
            dims=["components", "frames"],
            coords={"components": list(manager.component_ids), "frames": [0, 1]},
        )

        manager.populate_from_traces(new_traces)
        assert manager.traces.shape == (3, 7)  # Original 5 + 2 new frames

        # Test with unknown component
        invalid_traces = xr.DataArray(
            np.ones((1, 2)),
            dims=["components", "frames"],
            coords={"components": [999], "frames": [0, 1]},
        )
        with pytest.raises(
            ValueError, match="Cannot add traces for components that don't exist"
        ):
            manager.populate_from_traces(invalid_traces)

    def test_get_component(self, populated_manager):
        """Test getting components."""
        manager, components = populated_manager

        # Test getting existing component
        component = components[0]
        retrieved = manager.get_component(component.id)
        assert retrieved == component

        # Test getting non-existent component
        assert manager.get_component(999) is None

    def test_get_components_by_type(self, populated_manager):
        """Test getting components by type."""
        manager, _ = populated_manager

        neurons = manager.get_components_by_type(Neuron)
        backgrounds = manager.get_components_by_type(Background)

        assert len(neurons) == 2  # Components 0 and 2
        assert len(backgrounds) == 1  # Component 1

    def test_get_overlapping_components(self, populated_manager):
        """Test getting overlapping components."""
        manager, components = populated_manager

        # Components 0 and 1 should overlap due to their footprint patterns
        overlapping = manager.get_overlapping_components(components[0].id)
        assert components[1].id in overlapping

    def test_add_component(self, empty_manager, sample_footprint, sample_trace):
        """Test adding components."""
        component = Neuron(detected_frame_idx=0)
        empty_manager.add_component(component, sample_footprint, sample_trace)

        assert empty_manager.n_components == 1
        assert component.id in empty_manager.component_ids
        assert (
            empty_manager.footprints.sel(components=component.id)
            .drop_vars("components")
            .equals(sample_footprint)
        )
        assert (
            empty_manager.traces.sel(components=component.id)
            .drop_vars("components")
            .equals(sample_trace)
        )

    def test_remove_component(self, populated_manager):
        """Test removing components."""
        manager, components = populated_manager
        initial_count = manager.n_components
        component = components[0]

        removed = manager.remove_component(component.id)
        assert removed == component
        assert manager.n_components == initial_count - 1
        assert component.id not in manager.component_ids

    def test_update_component_timetrace(self, populated_manager, sample_trace):
        """Test updating component timetraces."""
        manager, components = populated_manager
        component = components[0]

        # Test successful update
        new_trace = sample_trace.copy()
        new_trace[:] = 10
        success = manager.update_component_timetrace(component.id, new_trace)
        assert success
        assert (
            manager.traces.sel(components=component.id)
            .drop_vars("components")
            .equals(new_trace)
        )

        # Test update for non-existent component
        assert not manager.update_component_timetrace(999, new_trace)

    def test_update_component_footprint(self, populated_manager, sample_footprint):
        """Test updating component footprints."""
        manager, components = populated_manager
        component = components[0]

        # Test successful update
        new_footprint = sample_footprint.copy()
        new_footprint[:] = 10
        success = manager.update_component_footprint(component.id, new_footprint)
        assert success
        assert (
            manager.footprints.sel(components=component.id)
            .drop_vars("components")
            .equals(new_footprint)
        )

        # Test update for non-existent component
        assert not manager.update_component_footprint(999, new_footprint)

    def test_get_time_traces_batch(self, populated_manager):
        """Test getting time trace batches."""
        manager, _ = populated_manager

        batch = manager.get_time_traces_batch(1, 3)
        assert isinstance(batch, xr.DataArray)
        assert batch.sizes["frames"] == 3
        assert set(batch.dims) == {"components", "frames"}

    def test_iterate_time_traces(self, populated_manager):
        """Test iterating over time traces."""
        manager, _ = populated_manager

        batches = list(manager.iterate_time_traces(batch_size=2))
        assert len(batches) == 3  # 5 frames with batch_size=2: [0,1], [2,3], [4]

        for start_idx, end_idx, batch in batches:
            assert isinstance(batch, xr.DataArray)
            assert batch.sizes["frames"] <= 2
            assert set(batch.dims) == {"components", "frames"}

    def test_neuron_ids(self, populated_manager):
        """Test getting neuron IDs."""
        manager, components = populated_manager
        neuron_ids = manager.neuron_ids
        assert len(neuron_ids) == 2
        assert all(isinstance(manager.get_component(id), Neuron) for id in neuron_ids)

    def test_background_ids(self, populated_manager):
        """Test getting background IDs."""
        manager, components = populated_manager
        background_ids = manager.background_ids
        assert len(background_ids) == 1
        assert all(
            isinstance(manager.get_component(id), Background) for id in background_ids
        )
