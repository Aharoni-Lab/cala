from typing import List

import numpy as np
import pytest
import xarray as xr

from cala.streaming.core.stores.base import BaseStore


class Store(BaseStore):
    """Concrete implementation of BaseStore for testing"""

    def temporal_update(self, last_streamed_data: xr.DataArray, ids: List[str]) -> None:
        pass


class TestBaseStore:
    @pytest.fixture
    def basic_store(self):
        return Store(
            dimensions=("component", "x", "y"), component_dimension="component"
        )

    @pytest.fixture
    def sample_data(self):
        return {
            "data": np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            "ids": ["id1", "id2"],
            "types": ["neuron", "background"],
        }

    def test_initialization(self, basic_store):
        """Test proper initialization of BaseStore"""
        assert basic_store.dimensions == ("component", "x", "y")
        assert basic_store.component_dimension == "component"
        assert basic_store.id_coordinate == "id_"
        assert basic_store.type_coordinate == "type_"
        assert isinstance(basic_store.warehouse, xr.DataArray)

    def test_invalid_component_dimension(self):
        """Test initialization with invalid component dimension"""
        with pytest.raises(ValueError):
            Store(dimensions=("x", "y"), component_dimension="component")

    def test_generate_store(self, basic_store, sample_data):
        """Test store generation with sample data"""
        result = basic_store.generate_warehouse(
            sample_data["data"], sample_data["ids"], sample_data["types"]
        )
        assert isinstance(result, xr.DataArray)
        assert result.dims == basic_store.dimensions
        assert (
            list(result.coords[basic_store.id_coordinate].values) == sample_data["ids"]
        )
        assert (
            list(result.coords[basic_store.type_coordinate].values)
            == sample_data["types"]
        )

    def test_insert(self, basic_store, sample_data):
        """Test insert functionality"""
        # Test inplace=True
        basic_store.insert(
            sample_data["data"], sample_data["ids"], sample_data["types"], inplace=True
        )
        assert len(basic_store.warehouse.coords[basic_store.id_coordinate]) == 2

        # Test inplace=False
        new_data = np.array([[[9, 10], [11, 12]]])
        result = basic_store.insert(new_data, ["id3"], ["neuropil"], inplace=False)
        assert isinstance(result, xr.DataArray)
        assert len(result.coords[basic_store.id_coordinate]) == 3

    def test_slice(self, basic_store, sample_data):
        """Test slice functionality"""
        basic_store.insert(
            sample_data["data"], sample_data["ids"], sample_data["types"], inplace=True
        )

        result = basic_store.slice(["id1"], ["neuron"])
        assert result.coords[basic_store.id_coordinate].values == "id1"

    def test_delete(self, basic_store, sample_data):
        """Test delete functionality

        should add more tests for multiple ids, and types that overlap / doesn't overlap etc.
        """
        basic_store.insert(
            sample_data["data"], sample_data["ids"], sample_data["types"], inplace=True
        )

        # Test inplace=False
        result = basic_store.delete(["id2"], ["background"], inplace=False)
        assert "id2" not in result.coords[basic_store.id_coordinate].values

        # Test inplace=True
        basic_store.delete(["id1"], ["neuron"], inplace=True)
        assert (
            "id1" not in basic_store.warehouse.coords[basic_store.id_coordinate].values
        )

    def test_update(self, basic_store, sample_data):
        """Test update functionality"""
        basic_store.insert(
            sample_data["data"], sample_data["ids"], sample_data["types"], inplace=True
        )

        update_data = basic_store.warehouse.copy()
        update_data.loc[{"id_": "id1"}] = np.ones((2, 2))

        # Test inplace=True
        basic_store.update(update_data, inplace=True)
        assert np.all(basic_store.warehouse.sel(id_="id1") == 1)

        # Test inplace=False
        result = basic_store.update(update_data, inplace=False)
        assert isinstance(result, xr.DataArray)

    def test_property_accessors(self, basic_store, sample_data):
        """Test property accessors"""
        basic_store.insert(
            sample_data["data"], sample_data["ids"], sample_data["types"], inplace=True
        )

        assert basic_store._types == sample_data["types"]
        assert basic_store._ids == sample_data["ids"]

        # Test id_to_type mapping
        id_type_map = basic_store.id_to_type
        assert id_type_map["id1"] == str
        assert id_type_map["id2"] == int

        # Test type_to_id mapping
        type_id_map = basic_store.type_to_id
        assert type_id_map[str] == ["id1"]
        assert type_id_map[int] == ["id2"]

    def test_where(self, basic_store, sample_data):
        """Test where functionality"""
        basic_store.insert(
            sample_data["data"], sample_data["ids"], sample_data["types"], inplace=True
        )

        condition = basic_store.warehouse > 5
        result = basic_store.where(condition, -1)
        assert isinstance(result, xr.DataArray)
        assert np.all(result.where(condition, -1).values[result.values <= 5] == -1)

    def test_warehouse_setter_validation(self, basic_store):
        """Test warehouse setter validation"""
        invalid_data = xr.DataArray(np.ones((2, 2)), dims=("x", "y"))

        with pytest.raises(ValueError):
            basic_store.warehouse = invalid_data
