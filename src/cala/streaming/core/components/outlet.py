from dataclasses import dataclass, field
from typing import List

import xarray as xr

from cala.streaming.core.components.registry import Registry
from cala.streaming.core.components.stores import FootprintStore, TraceStore
from cala.streaming.types.types import Footprints, Traces


@dataclass
class DataOutlet:
    """Manages a collection of fluorescent components (neurons and background)."""

    component_axis: str = "components"
    """The axis of the component."""
    spatial_axes: tuple = ("width", "height")
    """The spatial axes of the component."""
    frame_axis: str = "frames"
    """The axis of the frames."""

    registry: Registry = field(default_factory=Registry)
    footprints: FootprintStore = field(default_factory=lambda: FootprintStore())
    traces: TraceStore = field(default_factory=lambda: TraceStore())

    @property
    def _type_to_store(self):
        from .stores import BaseStore

        return {
            type(getattr(self, attr).array): attr
            for attr in dir(self)
            if isinstance(getattr(self, attr), BaseStore)
        }

    def __post_init__(self):
        # Ensure consistent axis names across managers
        self.footprints.component_axis = self.component_axis
        self.footprints.spatial_axes = self.spatial_axes
        self.traces.component_axis = self.component_axis
        self.traces.frame_axis = self.frame_axis

    def _assign(self, result: xr.DataArray | tuple[xr.DataArray, ...]):
        """Assign the entire matching store with the result.

        Args:
            result: Either a single DataArray or tuple of DataArrays to update state with
        """
        if len(self.registry.ids) > 0:
            raise AttributeError(
                "Cannot assign while cells exist. Use insert or replace."
            )
        results = (result,) if isinstance(result, xr.DataArray) else result

        # Assign to stores whose array category match result values
        for value in results:
            if store_name := self._type_to_store.get(type(value)):
                # are we updating a store, or overhauling a store? REMINDER: outlet has the record of ALL ids.
                # The following list covers all operations that can ACCEPT an array.✅🏗

                # scenario 1: cells already exist. new array with ids come in
                # (update) we're updating the observables of the existing cells

                # scenario 2: cells already exist. new array without ids come in
                # (register + insert) we're inserting newly-detected cells.

                # scenario 3: cells already exist. new array with partially filled ids comes in
                # (register + insert) + (update) new cells are detected AND existing cells are updated 🤔

                # scenario 4: cells do not exist. new array with ids come in
                # (load) we're loading saved data into a fresh outlet.

                # scenario 5: cells do not exist. new array without ids come in
                # (register + assign) make cells. assign them to array. save array 🏗

                # scenario 6: cells do not exist. new array with partially filled ids comes in
                # (🚫) should not be possible. loading data, found new cells without registering, and then trying to register...

                # scenario 7: cells already exist. new array with ids come in
                # (remove) --> we're removing cells that are missing in the incoming array

                # scenario 8: cells already exist. new array with partially filled ids come in
                # (remove) + (register + insert) update if exists in incoming. remove if missing in incoming. insert if no id. same as 10 but just happens to have no overlap 🤔

                # scenario 8: cells already exist. new sub-array with filled ids come in
                # (remove) + (update) update if exists in incoming. remove if missing in incoming. 🤔

                # scenario 9: cells already exist. new array with partially filled ids come in
                # (remove) + (register + insert) + (update) update if exists in incoming. remove if missing in incoming. insert if no id 🤔

                getattr(self, store_name).array = value

    def _insert(self, result: xr.DataArray | tuple[xr.DataArray, ...]):
        """Insert new component observables

        Args:
            result: Either a single DataArray or tuple of DataArrays to update state with
        """
        results = (result,) if isinstance(result, xr.DataArray) else result

        # Insert into stores whose array category match result values
        for value in results:
            if store_name := self._type_to_store.get(type(value)):
                getattr(self, store_name).insert(value)

    def register_footprints(self, footprint_batches: List[Footprints]) -> None:

        self.registry.clear()
        self.traces.array = Traces()

        assigned_batches = []
        for batch in footprint_batches:
            # get what components we have to generate
            component_type = (
                batch.__class__.__name__.lower()
            )  # - ish. this is going to give footprint rn. i need neuron, or background.
            # for each footprint, generate corresponding component type and collect the new component id
            component_ids = [
                self.registry.create(component_type)
                for _ in batch.coords[self.component_axis]
            ]

            batch.assign_coords({self.component_axis: component_ids})
            assigned_batches.append(batch)
        self.footprints.array = Footprints(
            xr.concat(assigned_batches, dim=self.component_axis)
        )
