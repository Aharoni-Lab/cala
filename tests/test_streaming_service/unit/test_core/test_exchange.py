from uuid import UUID

import numpy as np
import pytest

from cala.streaming.core.exchange import DataExchange
from cala.streaming.core.stores import FootprintStore, TraceStore
from cala.streaming.types import (
    FluorescentObject,
    Observable,
    NeuronFootprints,
    BackgroundFootprints,
)


# Mock classes for testing
class MockObservable(Observable):
    pass


class MockComponent(FluorescentObject):
    pass


class MockComposite(MockObservable, MockComponent):
    pass


class TestDataExchangeInitialization:
    def test_default_initialization(self):
        exchange = DataExchange()
        assert exchange.component_axis == "components"
        assert exchange.spatial_axes == ("width", "height")
        assert exchange.frame_axis == "frames"
        assert isinstance(exchange.footprints, FootprintStore)
        assert isinstance(exchange.traces, TraceStore)

    def test_custom_axes_initialization(self):
        exchange = DataExchange(
            component_axis="cells", spatial_axes=("x", "y"), frame_axis="time"
        )
        assert exchange.component_axis == "cells"
        assert exchange.spatial_axes == ("x", "y")
        assert exchange.frame_axis == "time"


class TestDataExchangeTypeHandling:
    @pytest.fixture
    def exchange(self):
        return DataExchange()

    def test_find_intersection_type_success(self, exchange):
        result = exchange._find_intersection_type_of(Observable, MockObservable())
        assert result == MockObservable

    def test_find_intersection_type_failure(self, exchange):
        class UnrelatedClass:
            pass

        with pytest.raises(TypeError):
            exchange._find_intersection_type_of(Observable, UnrelatedClass())

    def test_get_observable_x_component(self, exchange):
        with pytest.raises(KeyError):
            exchange.get_observable_x_component(MockComposite)

    def test_get_observable_x_component_invalid_type(self, exchange):
        class InvalidType:
            pass

        with pytest.raises(TypeError):
            exchange.get_observable_x_component(InvalidType)


class TestDataExchangeCollection:
    @pytest.fixture
    def exchange(self):
        return DataExchange()

    @pytest.fixture
    def mock_data_array(self):
        data = np.random.rand(3, 10, 10)  # 3 components, 10x10 spatial dimensions
        return NeuronFootprints(
            data,
            dims=["components", "width", "height"],
            coords={"components": [0, 1, 2]},
        )

    def test_collect_unregistered_components(self, exchange, mock_data_array):
        # Test collecting new components
        exchange.collect(mock_data_array)
        # Verify that UUIDs were assigned
        assert all(
            isinstance(comp_id, UUID)
            for comp_id in exchange.footprints.warehouse.coords["id_"].values
        )

    def test_collect_registered_components(self, exchange, mock_data_array):
        # First collection to register components
        exchange.collect(mock_data_array)
        original_ids = exchange.footprints.warehouse.coords["id_"].values

        # Second collection with same data
        exchange.collect(
            BackgroundFootprints(
                data=np.ones_like(exchange.footprints.warehouse),
                dims=["components", "width", "height"],
                coords={"components": [0, 1, 2]},
            )
        )

        # Verify IDs are accumulated
        assert set(original_ids).issubset(
            set(exchange.footprints.warehouse.coords["id_"].values)
        )
        assert exchange.footprints.warehouse.sizes["components"] == 6

    def test_collect_multiple_arrays(self, exchange, mock_data_array):
        # Test collecting multiple arrays at once
        second_array = mock_data_array.copy()
        exchange.collect((mock_data_array, second_array))
        assert exchange.footprints.warehouse.sizes["components"] == 6

    def test_collect_get_observable_x_component(self, exchange, mock_data_array):
        exchange.collect(mock_data_array)
        # Second collection with same data
        exchange.collect(
            BackgroundFootprints(
                data=np.ones_like(exchange.footprints.warehouse),
                dims=["components", "width", "height"],
                coords={"components": [0, 1, 2]},
            )
        )
        result = exchange.get_observable_x_component(NeuronFootprints)
        assert set(result.coords[exchange.type_coord].values.tolist()) == {"Neuron"}


class TestDataExchangeEdgeCases:
    @pytest.fixture
    def exchange(self):
        return DataExchange()

    def test_collect_empty_array(self, exchange):
        empty_array = NeuronFootprints(
            np.array([]).reshape(0, 10, 10),
            dims=["components", "width", "height"],
            coords={"components": []},
        )
        exchange.collect(empty_array)
        assert exchange.footprints.warehouse is not None

    def test_type_to_store_property(self, exchange):
        store_mapping = exchange.type_to_store
        assert isinstance(store_mapping, dict)
        assert all(issubclass(key, Observable) for key in store_mapping.keys())
        assert all(isinstance(value, str) for value in store_mapping.values())
