from dataclasses import dataclass, field
from typing import Any, Dict, Type

import xarray as xr

from cala.streaming.core.components.registry import Registry
from cala.streaming.core.components.stores import FootprintStore, TraceStore
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

    registry: Registry = field(default_factory=Registry)
    footprints: FootprintStore = field(default_factory=lambda: FootprintStore())
    traces: TraceStore = field(default_factory=lambda: TraceStore())

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

        type_ids = self.registry.get_id_by_type(component_type)
        return getattr(self, self.type_to_store[observable_type]).array.sel(
            {self.component_axis: type_ids}
        )

    @property
    def type_to_store(self) -> Dict[Type["Observable"], str]:
        from .stores import BaseStore

        return {
            type(attr_class().array): attr
            for attr, attr_class in self.__annotations__.items()
            if isinstance(getattr(self, attr), BaseStore)
        }

    def __post_init__(self):
        # Ensure consistent axis names across managers
        self.footprints.component_axis = self.component_axis
        self.footprints.spatial_axes = self.spatial_axes
        self.traces.component_axis = self.component_axis
        self.traces.frame_axis = self.frame_axis

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

    def collect(self, result: xr.DataArray | tuple[xr.DataArray, ...]) -> None:
        # Init steps: assign -> insert -> assign -> insert
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

            ids = self.registry.create_many(
                value.sizes[self.component_axis], component_type
            )
            value = value.assign_coords({self.component_axis: ids})

            if store_name := self.type_to_store.get(observable_type):

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
