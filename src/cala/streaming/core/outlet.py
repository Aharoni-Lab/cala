from dataclasses import dataclass, field
from typing import Any, Dict, Type, List
from uuid import uuid4

import xarray as xr

from cala.streaming.core.stores import FootprintStore, TraceStore
from cala.streaming.types import FluorescentObject, Observable


@dataclass
class DataOutlet:
    """Manages a collection of fluorescent components (neurons and background)."""

    # CRUD: are we updating a store, or overhauling a store?
    # REMINDER: outlet has the record of ALL ids.
    # REMINDER: Now we are type safe!!!

    # Init steps: register -> assign -> insert -> assign -> insert

    # The following list covers all operations that can ACCEPT an array. (✅🏗)

    # DURING INIT: it's all different. i might need a separate state for init stage.

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

    ids: List[str] = field(default_factory=list)  # idk if i need this

    footprints: FootprintStore = field(init=False)
    traces: TraceStore = field(init=False)

    def __post_init__(self):
        # Ensure consistent axis names across stores
        self.footprints = FootprintStore(...)
        self.traces = TraceStore(...)

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
            types=component_type
        )

    @property
    def type_to_store(self) -> Dict[Type["Observable"], str]:
        from .stores import BaseStore

        return {
            type(attr_class().array): attr
            for attr, attr_class in self.__annotations__.items()
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
        return components.coords[self.component_axis].dtype == int

    def _is_ids_subset(self, ids: List[str]) -> bool:
        return set(ids).issubset(set(self.ids))

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
                # but some ids are foreign
                if not self._is_ids_subset(value.ids):
                    raise ValueError(
                        "Some incoming components with an ID are not in the registry book."
                    )
            # not registered yet
            else:
                ids = [uuid4().hex for _ in range(value.sizes[self.component_axis])]
                types = [component_type] * value.sizes[self.component_axis]

            # determine which store to input the value into
            if store_name := self.type_to_store.get(observable_type):

                # if the store is empty, we assign instead of insert
                if len(getattr(self, store_name).array.sizes) == 0:
                    getattr(self, store_name).array = value
                else:
                    getattr(self, store_name).insert(value)

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
