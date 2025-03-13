from dataclasses import dataclass, field
from typing import Dict, Type

import xarray as xr

from cala.streaming.stores.common import FootprintStore, TraceStore
from cala.streaming.types import Observable
from cala.streaming.types.common import find_intersection_type_of


@dataclass
class DataExchange:
    """Manages a collection of fluorescent components (neurons and background)."""

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

    def get_type(self, type_: Type) -> Observable:
        # Test what happens when the composite type is a member of none.
        observable_type = find_intersection_type_of(
            base_type=Observable, instance=type_
        )

        if not observable_type:
            raise TypeError(
                f"The provided type {type_} is not a composite type of Observable and FluorescentObject"
            )

        return getattr(self, self.type_to_store[observable_type]).get_type(type_=type_)

    @property
    def type_to_store(self) -> Dict[Type["Observable"], str]:
        from .stores import BaseStore

        return {
            getattr(self, attr).data_type: attr
            for attr in self.__annotations__.keys()
            if isinstance(getattr(self, attr), BaseStore)
        }

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
                observable_type = find_intersection_type_of(
                    base_type=Observable, instance=value
                )

            except TypeError:
                continue

            # determine which store to input the value into
            if store_name := self.type_to_store.get(observable_type):
                getattr(self, store_name).insert(value, inplace=True)
