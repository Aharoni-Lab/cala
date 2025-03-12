from dataclasses import dataclass, field
from typing import Any, Dict, Type
from uuid import uuid4

import xarray as xr

from cala.streaming.core.stores import FootprintStore, TraceStore
from cala.streaming.types import FluorescentObject, Observable


@dataclass
class DataExchange:
    """Manages a collection of fluorescent components (neurons and background)."""

    # CRUD: are we updating a store, or overhauling a store?
    # REMINDER: outlet has the record of ALL ids.
    # REMINDER: Now we are type safe!!!

    # Init steps: assign -> insert -> assign -> insert

    # The following list covers all operations that can ACCEPT an array. (✅🏗)

    # scenario 4: cells do not exist. new array with ids come in
    # (load) we're loading saved data into a fresh outlet.

    # scenario 5: cells do not exist. new array without ids come in
    # INIT: (register ✅ + assign ✅) make cells. assign them to array. save array 🏗

    # scenario 6: cells do not exist. new array with partially filled ids comes in
    # (🚫) should not be possible. loading data, found new cells without registering, and then trying to register...

    # scenario 1: cells already exist. new array with ids come in
    # (update) we're updating the observables of the existing cells

    # scenario 2: cells already exist. new array without ids come in
    # (register ✅ + insert ✅) we're inserting newly-detected cells.

    # scenario 3: cells already exist. new array with partially filled ids comes in
    # (register + insert) + (update) new cells are detected AND existing cells are updated 🤔

    # scenario 7: cells already exist. new array with ids come in
    # (remove) --> we're removing cells that are missing in the incoming array

    # scenario 8: cells already exist. new array with partially filled ids come in
    # (remove) + (register + insert) update if exists in incoming. remove if missing in incoming. insert if no id. same as 10 but just happens to have no overlap 🤔

    # scenario 8: cells already exist. new sub-array with filled ids come in
    # (remove) + (update) update if exists in incoming. remove if missing in incoming. 🤔

    # scenario 9: cells already exist. new array with partially filled ids come in
    # (remove) + (register + insert) + (update) update if exists in incoming. remove if missing in incoming. insert if no id 🤔

    component_axis: str = "components"
    """The axis of the component."""
    spatial_axes: tuple = ("width", "height")
    """The spatial axes of the component."""
    frame_axis: str = "frames"
    """The axis of the frames."""

    id_coord: str = "id_"
    type_coord: str = "type_"

    footprints: FootprintStore = field(init=False)
    traces: TraceStore = field(init=False)

    def __post_init__(self):
        self.footprints = FootprintStore(
            dimensions=(self.component_axis, *self.spatial_axes),
            component_dim=self.component_axis,
            spatial_axes=self.spatial_axes,
            id_coord=self.id_coord,
            type_coord=self.type_coord,
        )
        self.traces = TraceStore(
            dimensions=(self.component_axis, self.frame_axis),
            component_dim=self.component_axis,
            frame_axis=self.frame_axis,
            id_coord=self.id_coord,
            type_coord=self.type_coord,
        )

    def get_observable_x_component(self, composite_type: Type) -> Observable:
        # Test what happens when the composite type is a member of none.
        observable_type = self._find_intersection_type_of(
            base_type=Observable, instance=composite_type
        )
        component_type = self._find_intersection_type_of(
            base_type=FluorescentObject, instance=composite_type
        )

        if not all([observable_type, component_type]):
            raise TypeError(
                f"The provided type {composite_type} is not a composite type of Observable and FluorescentObject"
            )

        return getattr(self, self.type_to_store[observable_type]).slice(
            types=[component_type.__name__]
        )

    @property
    def type_to_store(self) -> Dict[Type["Observable"], str]:
        from .stores import BaseStore

        return {
            getattr(self, attr).data_type: attr
            for attr in self.__annotations__.keys()
            if isinstance(getattr(self, attr), BaseStore)
        }

    @staticmethod
    def _find_intersection_type_of(base_type: Type, instance: Any) -> Type:
        component_types = set(base_type.__subclasses__())
        try:
            parent_types = set(instance.__bases__)
        except AttributeError:
            try:
                return {
                    component_type
                    for component_type in component_types
                    if isinstance(instance, component_type)
                }.pop()
            except KeyError:
                raise TypeError(
                    f"The instance type {instance.__class__.__name__} does not inherit from one of the component types: {component_types}"
                )

        component_type = parent_types & component_types
        if len(component_type) != 1:
            raise TypeError(
                f"The instance type {instance.__class__.__name__} does not inherit from one of the component types: {component_types}"
            )
        return component_type.pop()

    def _is_unregistered(self, components: xr.DataArray) -> bool:
        return len([k for k, v in components.coords.items() if k == self.id_coord]) == 0

    def collect(self, result: xr.DataArray | tuple[xr.DataArray, ...]) -> None:
        # Init steps: assign1 -> insert2 -> assign3 -> insert4
        # assign1: these are new cells, there are no arrays. --> so we create cells and assign
        # insert2: these are new cells, there is a same type array. -->
        # assign3: these are existing cells, there is a same type array.
        # insert4: these are existing cells, there is a same type array.

        # checking new cell status: check if coords is hex or int! hex: existing, int: new
        # then, all we need is if already exists --> insert, if does not exist --> assign

        results = (result,) if isinstance(result, xr.DataArray) else result

        for value in results:
            try:
                observable_type = self._find_intersection_type_of(
                    base_type=Observable, instance=value
                )
                component_type = self._find_intersection_type_of(
                    base_type=FluorescentObject, instance=value
                )
            except TypeError:
                continue

            # registered
            if not self._is_unregistered(value):
                # determine which store to input the value into
                if store_name := self.type_to_store.get(observable_type):
                    getattr(self, store_name).insert(value, inplace=True)
            # not registered yet
            else:
                ids = [uuid4() for _ in range(value.sizes[self.component_axis])]
                types = [component_type.__name__] * value.sizes[self.component_axis]
                # determine which store to input the value into
                if store_name := self.type_to_store.get(observable_type):
                    value = getattr(self, store_name).generate_warehouse(
                        value, ids, types
                    )
                    getattr(self, store_name).insert(value, inplace=True)

    def _assign(self, result: xr.DataArray | tuple[xr.DataArray, ...]):
        """Assign the entire matching store with the result.

        Args:
            result: Either a single DataArray or tuple of DataArrays to update state with
        """
        results = (result,) if isinstance(result, xr.DataArray) else result

        # Assign to stores whose array category match result values
        for value in results:
            if store_name := self.type_to_store.get(
                self._find_intersection_type_of(base_type=Observable, instance=value)
            ):

                getattr(self, store_name).array = value

    def _insert(self, result: xr.DataArray | tuple[xr.DataArray, ...]):
        """Insert new component observables

        Args:
            result: Either a single DataArray or tuple of DataArrays to update state with
        """
        results = (result,) if isinstance(result, xr.DataArray) else result

        # Insert into stores whose array category match result values
        for value in results:
            if store_name := self.type_to_store.get(
                self._find_intersection_type_of(base_type=Observable, instance=value)
            ):
                getattr(self, store_name).insert(value)
