import numpy as np
import pytest
import xarray as xr

from cala.streaming.core.components.stores.traces import TraceStore


class TestTraceManager:
    @pytest.fixture
    def basic_manager(self):
        """Create a basic TraceManager for testing."""
        return TraceStore()

    @pytest.fixture
    def initialized_manager(self, basic_manager):
        """Create a TraceManager initialized with empty traces."""
        component_ids = [1, 2, 3]
        basic_manager.initialize(component_ids)
        return basic_manager

    @pytest.fixture
    def sample_trace(self):
        """Create a sample trace for testing."""
        data = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        return xr.DataArray(data, dims=["frames"], coords={"frames": range(len(data))})

    def test_initialization(self, basic_manager):
        """Test basic initialization of TraceManager."""
        assert basic_manager.component_axis == "component"
        assert basic_manager.frame_axis == "frames"
        assert basic_manager.dimensions == ("component", "frames")

    def test_initialize_with_components(self, basic_manager):
        """Test initializing with component IDs."""
        component_ids = [1, 2, 3]
        basic_manager.initialize(component_ids)

        assert hasattr(basic_manager, "_traces")
        assert isinstance(basic_manager.traces, xr.DataArray)
        assert set(basic_manager.traces.dims) == {"component", "frames"}
        assert list(basic_manager.traces.component.values) == component_ids
        assert basic_manager.traces.sizes["frames"] == 0

    def test_add_empty_trace(self, initialized_manager):
        """Test adding an empty trace."""
        initialized_manager.insert(4)  # New component ID

        assert 4 in initialized_manager.traces.component.values
        assert len(initialized_manager.traces.component) == 4
        np.testing.assert_array_equal(
            initialized_manager.traces.sel(component=4).values,
            np.zeros(initialized_manager.traces.sizes["frames"]),
        )

    def test_add_trace_with_data(self, initialized_manager, sample_trace):
        """Test adding a trace with data."""
        initialized_manager.insert(4, sample_trace)

        assert 4 in initialized_manager.traces.component.values
        np.testing.assert_array_equal(
            initialized_manager.traces.sel(component=4).values, sample_trace.values
        )

    def test_remove_trace(self, initialized_manager):
        """Test removing a trace."""
        initialized_manager.remove(2)

        assert 2 not in initialized_manager.traces.component.values
        assert len(initialized_manager.traces.component) == 2
        assert list(initialized_manager.traces.component.values) == [1, 3]

    def test_update_trace(self, initialized_manager, sample_trace):
        """Test updating an existing trace."""
        # First append some frames so we have data to update
        new_traces = xr.DataArray(
            np.ones((3, 5)),  # 3 components, 5 frames
            coords={"component": [1, 2, 3], "frames": range(5)},
            dims=("component", "frames"),
        )
        initialized_manager.append(new_traces)

        # Now update one component's trace
        initialized_manager.replace(2, sample_trace)
        np.testing.assert_array_equal(
            initialized_manager.traces.sel(component=2).values, sample_trace.values
        )

    def test_update_trace_wrong_shape(self, initialized_manager, sample_trace):
        """Test updating with wrong shape raises error."""
        wrong_shape_trace = xr.DataArray(
            np.ones(3), dims=["frames"], coords={"frames": range(3)}
        )
        with pytest.raises(
            ValueError, match="New trace shape doesn't match existing trace"
        ):
            initialized_manager.replace(1, wrong_shape_trace)

    def test_append_frames(self, initialized_manager):
        """Test appending new frames."""
        # Create new traces for some components
        new_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2 components, 2 frames
        new_traces = xr.DataArray(
            new_data,
            coords={"component": [1, 2], "frames": [0, 1]},
            dims=("component", "frames"),
        )

        initialized_manager.append(new_traces)

        # Check dimensions
        assert initialized_manager.traces.sizes["frames"] == 2
        # Check values for components with explicit traces
        np.testing.assert_array_equal(
            initialized_manager.traces.sel(component=[1, 2]).values, new_data
        )
        # Check values for component without explicit trace (should be zeros)
        np.testing.assert_array_equal(
            initialized_manager.traces.sel(component=3).values, np.zeros(2)
        )

    def test_append_frames_wrong_dimensions(self, initialized_manager):
        """Test appending frames with wrong dimensions raises error."""
        wrong_dims = xr.DataArray(np.ones((2, 3)), dims=["wrong", "dims"])
        with pytest.raises(ValueError, match="Traces dimensions must be"):
            initialized_manager.append(wrong_dims)

    def test_get_batch(self, initialized_manager):
        """Test getting a batch of traces."""
        # First append some frames
        new_traces = xr.DataArray(
            np.arange(15).reshape(3, 5),  # 3 components, 5 frames
            coords={"component": [1, 2, 3], "frames": range(5)},
            dims=("component", "frames"),
        )
        initialized_manager.append(new_traces)

        # Get a batch
        batch = initialized_manager.get_batch(1, 3)
        assert batch.sizes["frames"] == 3  # frames 1, 2, 3
        np.testing.assert_array_equal(
            batch.values, np.arange(15).reshape(3, 5)[:, 1:4]  # Select frames 1-3
        )

        # Test inclusive end
        batch = initialized_manager.get_batch(0, 0)
        assert batch.sizes["frames"] == 1  # Just frame 0
        np.testing.assert_array_equal(batch.values, new_traces.values[:, 0:1])

    def test_iterate_batches(self, initialized_manager):
        """Test iterating over batches."""
        # First append some frames
        new_traces = xr.DataArray(
            np.arange(30).reshape(3, 10),  # 3 components, 10 frames
            coords={"component": [1, 2, 3], "frames": range(10)},
            dims=("component", "frames"),
        )
        initialized_manager.append(new_traces)

        # Test iteration with batch_size=4
        batches = list(initialized_manager.iterate_batches(batch_size=4))
        assert len(batches) == 3  # Should get 3 batches (4+4+2 frames)

        # Check first batch
        start, end, batch = batches[0]
        assert start == 0 and end == 3  # Inclusive end
        assert batch.sizes["frames"] == 4  # frames 0,1,2,3
        np.testing.assert_array_equal(batch.values, new_traces.values[:, 0:4])

        # Check last batch
        start, end, batch = batches[-1]
        assert start == 8 and end == 9  # Inclusive end
        assert batch.sizes["frames"] == 2  # frames 8,9
        np.testing.assert_array_equal(batch.values, new_traces.values[:, 8:10])

    def test_iterate_batches_edge_cases(self, initialized_manager):
        """Test batch iteration edge cases."""
        # Test empty traces
        batches = list(initialized_manager.iterate_batches(batch_size=4))
        assert len(batches) == 0

        # Test invalid batch size
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(initialized_manager.iterate_batches(batch_size=0))
        with pytest.raises(ValueError, match="batch_size must be positive"):
            list(initialized_manager.iterate_batches(batch_size=-1))

        # Test single frame
        single_frame = xr.DataArray(
            np.ones((3, 1)),
            coords={"component": [1, 2, 3], "frames": [0]},
            dims=("component", "frames"),
        )
        initialized_manager.append(single_frame)
        batches = list(initialized_manager.iterate_batches(batch_size=4))
        assert len(batches) == 1
        start, end, batch = batches[0]
        assert start == 0 and end == 0
        assert batch.sizes["frames"] == 1

        # Test batch size larger than total frames
        initialized_manager.append(
            xr.DataArray(
                np.ones((3, 2)),
                coords={"component": [1, 2, 3], "frames": [1, 2]},
                dims=("component", "frames"),
            )
        )
        batches = list(initialized_manager.iterate_batches(batch_size=10))
        assert len(batches) == 1
        start, end, batch = batches[0]
        assert start == 0 and end == 2
        assert batch.sizes["frames"] == 3
