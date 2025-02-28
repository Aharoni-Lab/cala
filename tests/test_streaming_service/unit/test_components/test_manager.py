import numpy as np
import pytest
from scipy import sparse

from cala.streaming.core.components.manager import ComponentManager
from cala.streaming.core.components.types import Neuron, Background


class TestComponentManager:
    @pytest.fixture
    def empty_manager(self):
        """Create an empty ComponentManager."""
        return ComponentManager()

    @pytest.fixture
    def multi_component_manager(self):
        """Create a ComponentManager with multiple components for ordering tests."""
        manager = ComponentManager()

        # Create components with distinct patterns for easy identification
        components = []
        for i in range(3):
            # Create footprint with unique pattern
            footprint = np.zeros((5, 5))
            footprint[i : i + 2, i : i + 2] = i + 1

            # Create time trace with unique pattern
            time_trace = np.array([float(i + 1)] * 4)  # [1,1,1,1], [2,2,2,2], [3,3,3,3]

            if i % 2 == 0:
                component = Neuron(footprint, time_trace)
            else:
                component = Background(footprint, time_trace)
            components.append(component)
            manager.add_component(component)

        return manager, components

    def test_initialization(self, empty_manager):
        """Test initialization of ComponentManager."""
        assert empty_manager.n_components == 0
        assert empty_manager._footprint_shape is None
        assert empty_manager._n_frames is None
        assert isinstance(empty_manager.footprints, np.ndarray)
        assert isinstance(empty_manager.time_traces, np.ndarray)
        assert len(empty_manager.time_traces) == 0

    def test_add_component(self, empty_manager):
        """Test adding components."""
        footprint = np.zeros((5, 5))
        footprint[0:2, 0:2] = 1
        time_trace = np.array([1.0, 2.0, 3.0, 4.0])
        neuron = Neuron(footprint, time_trace)

        empty_manager.add_component(neuron)
        assert empty_manager.n_components == 1
        assert empty_manager._footprint_shape == (5, 5)
        assert empty_manager._n_frames == 4

        # Test adding component with wrong shape
        wrong_footprint = np.zeros((6, 6))
        wrong_neuron = Neuron(wrong_footprint, time_trace)
        with pytest.raises(ValueError):
            empty_manager.add_component(wrong_neuron)

        # Test adding component with wrong time trace length
        wrong_trace = np.array([1.0, 2.0])
        wrong_neuron = Neuron(footprint, wrong_trace)
        with pytest.raises(ValueError):
            empty_manager.add_component(wrong_neuron)

    def test_remove_component(self, multi_component_manager):
        """Test removing components."""
        manager, components = multi_component_manager
        initial_count = manager.n_components
        removed = manager.remove_component(0)

        assert manager.n_components == initial_count - 1
        assert isinstance(removed, Neuron)

        # Check that the component was removed from overlapping sets
        remaining_component = manager.get_component(0)
        assert remaining_component is not None
        assert removed not in remaining_component.overlapping_objects

    def test_get_component(self, multi_component_manager):
        """Test getting components by index."""
        manager, components = multi_component_manager

        # First component should be a Neuron
        component = manager.get_component(0)
        assert isinstance(component, Neuron)

        # Second component should be a Background
        component = manager.get_component(1)
        assert isinstance(component, Background)

        # Test invalid index
        assert manager.get_component(999) is None

    def test_update_component_timetrace(self, multi_component_manager):
        """Test updating component time traces."""
        manager, components = multi_component_manager
        new_trace = np.array([4.0, 5.0, 6.0, 7.0])
        success = manager.update_component_timetrace(0, new_trace)

        assert success
        component = manager.get_component(0)
        assert np.array_equal(component.time_trace[:], new_trace)

    def test_update_component_footprint(self, multi_component_manager):
        """Test updating component footprints."""
        manager, components = multi_component_manager
        new_footprint = np.zeros((5, 5))
        new_footprint[1:3, 1:3] = 1

        success = manager.update_component_footprint(0, new_footprint)
        assert success

        component = manager.get_component(0)
        assert np.array_equal(component.footprint.toarray(), new_footprint)

        # Test wrong shape
        wrong_footprint = np.zeros((6, 6))
        with pytest.raises(ValueError):
            manager.update_component_footprint(0, wrong_footprint)

    def test_get_components_by_type(self, multi_component_manager):
        """Test getting components by type."""
        manager, components = multi_component_manager
        neurons = manager.get_components_by_type(Neuron)
        backgrounds = manager.get_components_by_type(Background)

        assert len(neurons) == 2  # Components 0 and 2 are neurons
        assert len(backgrounds) == 1  # Component 1 is background
        assert all(isinstance(n, Neuron) for n in neurons)
        assert all(isinstance(b, Background) for b in backgrounds)

    def test_neuron_and_background_indices(self, multi_component_manager):
        """Test getting neuron and background indices."""
        manager, components = multi_component_manager
        neuron_idx = manager.neuron_indices
        background_idx = manager.background_indices

        assert neuron_idx == [0, 2]  # First and third components are neurons
        assert background_idx == [1]  # Second component is background

    def test_footprints_property(self, multi_component_manager):
        """Test getting concatenated footprints."""
        manager, components = multi_component_manager
        footprints = manager.footprints
        assert isinstance(footprints, np.ndarray)
        assert footprints.shape == (3, 5, 5)  # 3 components, 5x5 shape

    def test_time_traces_property(self, multi_component_manager):
        """Test getting concatenated time traces."""
        manager, components = multi_component_manager
        traces = manager.time_traces
        assert isinstance(traces, np.ndarray)
        assert traces.shape == (3, 4)  # 3 components, 4 time points

    def test_get_time_traces_batch(self, multi_component_manager):
        """Test getting time trace batches."""
        manager, components = multi_component_manager
        batch = manager.get_time_traces_batch(0, 2)
        assert isinstance(batch, np.ndarray)
        assert batch.shape == (3, 2)  # 3 components, 2 time points

        # Test empty batch
        empty_batch = manager.get_time_traces_batch(10, 11)
        assert isinstance(empty_batch, np.ndarray)
        assert empty_batch.size == 0

    def test_iterate_time_traces(self, multi_component_manager):
        """Test iterating over time traces."""
        manager, components = multi_component_manager
        for start_idx, end_idx, batch in manager.iterate_time_traces(batch_size=2):
            assert isinstance(batch, np.ndarray)
            assert batch.shape[0] == 3  # 3 components
            assert batch.shape[1] <= 2  # batch_size or less

    def test_empty_manager_properties(self, empty_manager):
        """Test properties of empty manager."""
        assert empty_manager.n_components == 0
        assert len(empty_manager.neuron_indices) == 0
        assert len(empty_manager.background_indices) == 0
        assert empty_manager.footprints.shape == (
            3,
            2,
        )  # Default shape for empty manager
        assert empty_manager.time_traces.size == 0

    def test_check_overlap_static_method(self):
        """Test the static method for checking overlap."""
        # Create two overlapping sparse matrices
        footprint1 = sparse.csr_matrix(([1], ([2], [2])), shape=(5, 5))
        footprint2 = sparse.csr_matrix(([1], ([2], [2])), shape=(5, 5))

        # Test overlapping case
        assert ComponentManager._check_overlap(footprint1, footprint2)

        # Test non-overlapping case
        footprint3 = sparse.csr_matrix(([1], ([0], [0])), shape=(5, 5))
        assert not ComponentManager._check_overlap(footprint1, footprint3)

    def test_component_type_indices_consistency(self, multi_component_manager):
        """Test that neuron and background indices maintain consistency with component types."""
        manager, components = multi_component_manager

        # Get indices
        neuron_indices = manager.neuron_indices
        background_indices = manager.background_indices

        # Verify neuron indices
        for idx in neuron_indices:
            assert isinstance(manager._components[idx], Neuron)

        # Verify background indices
        for idx in background_indices:
            assert isinstance(manager._components[idx], Background)

        # Verify all components are accounted for
        assert len(neuron_indices) + len(background_indices) == len(manager._components)
        assert sorted(neuron_indices + background_indices) == list(
            range(len(manager._components))
        )

    @pytest.mark.parametrize(
        "operation", ["initial", "update_trace", "update_footprint", "remove", "add"]
    )
    def test_component_order_consistency(self, multi_component_manager, operation):
        """Test that component order remains consistent through various operations."""
        manager, original_components = multi_component_manager

        def verify_order(unmodified: bool = False):
            """Helper to verify component order consistency."""

            saved_ids = [component._id for component in manager._components]

            saved_footprint = [
                component.footprint.toarray() for component in manager._components
            ]
            retrieved_footprints = manager.footprints

            saved_traces = [
                component.time_trace[:] for component in manager._components
            ]
            retrieved_traces = manager.time_traces

            # Verify array lengths match
            assert (
                len(saved_ids)
                == len(saved_footprint)
                == len(retrieved_footprints)
                == len(saved_traces)
                == len(retrieved_traces)
            )
            # Verify component data matches in all arrays
            assert np.array_equal(saved_footprint, retrieved_footprints)
            assert np.array_equal(saved_traces, retrieved_traces)

            for idx, saved_trace in enumerate(saved_traces):
                # Verify batch access maintains order
                batch = manager.get_time_traces_batch(0, 1)
                assert np.array_equal(batch[idx][0], saved_trace[0])

            if unmodified:
                original_ids = [component._id for component in original_components]
                original_footprints = [
                    component.footprint.toarray() for component in original_components
                ]
                original_traces = [
                    component.time_trace[:] for component in original_components
                ]
                assert (
                    len(original_ids)
                    == len(original_footprints)
                    == len(original_traces)
                    == len(saved_ids)
                )
                assert np.array_equal(original_ids, saved_ids)
                assert np.array_equal(original_footprints, saved_footprint)
                assert np.array_equal(original_traces, saved_traces)

        # Initial state verification
        if operation == "initial":
            verify_order(unmodified=True)
            return

        # Apply the specified operation
        if operation == "update_trace":
            manager.update_component_timetrace(1, np.array([10.0, 10.0, 10.0, 10.0]))
        elif operation == "update_footprint":
            manager.update_component_footprint(0, np.ones((5, 5)))
        elif operation == "remove":
            manager.remove_component(1)
        elif operation == "add":
            new_footprint = np.zeros((5, 5))
            new_footprint[0:2, 0:2] = 2
            new_trace = np.array([2.0, 2.0, 2.0, 2.0])
            manager.add_component(Neuron(new_footprint, new_trace))

        verify_order()


# ! WE ALSO NEED BATCH MATRIX ADDITION/REMOVAL/UPDATE/MERGE TESTS
