import logging
from typing import Annotated

import numpy as np
import xarray as xr

from cala.streaming.core import Axis, ObservableStore

logger = logging.getLogger(__name__)


class FootprintStore(ObservableStore):
    """Spatial footprints of identified components.

    Represents the spatial distribution patterns of components (neurons or background)
    in the field of view. Each footprint typically contains the spatial extent and
    intensity weights of a component.
    """

    def update(self, data: xr.DataArray) -> None:
        if len(data) == 0:
            return None

        existing_ids = set(data.coords[Axis.id_coordinates].values) & set(
            self.warehouse.coords[Axis.id_coordinates].values
        )
        new_ids = set(data.coords[Axis.id_coordinates].values) - set(
            self.warehouse.coords[Axis.id_coordinates].values
        )

        if existing_ids and new_ids:  # detect returned the original
            raise NotImplementedError(
                "There should not be a case of both existing trace update and new components detection in update"
            )
        elif existing_ids:  # new frame footprint update
            self.warehouse = data
        elif new_ids:  # detect only returned new elements
            self.warehouse = xr.concat(
                [self.warehouse, data],
                dim=Axis.component_axis,
            )
        return None


Footprints = Annotated[xr.DataArray, FootprintStore]


class TraceStore(ObservableStore):
    """Temporal activity traces of identified components.

    Contains the time-varying fluorescence signals of components across frames,
    representing their activity patterns over time.
    """

    persistent = True

    @property
    def warehouse(self) -> xr.DataArray:
        return (
            xr.open_zarr(self.store_path)
            .isel({Axis.frames_axis: slice(-self.peek_size, None)})
            .to_dataarray()
            .isel({"variable": 0})  # not sure why it automatically makes this coordinate
            .reset_coords("variable", drop=True)
        )

    @warehouse.setter
    def warehouse(self, value: xr.DataArray) -> None:
        value.to_zarr(self.store_path, mode="w")  # need to make sure it can overwrite

    def _append(self, data: xr.DataArray, append_dim: str | list[str]) -> None:
        data.to_zarr(self.store_path, append_dim=append_dim)

    def update(self, data: xr.DataArray) -> None:
        # 4 possibilities:
        # 1. updating traces of existing items: (identical ids)
        # (a) one frame
        # (b) multiple frames
        # 2. detected new items: (new ids)
        # (a) one item
        # (b) multiple items
        # are we making copies?? yes we are. there's no other way, unfortunately.
        # https://stackoverflow.com/questions/33435953/is-it-possible-to-append-to-an-xarray-dataset

        if len(data) == 0:
            return

        warehouse_coords = self.warehouse.coords

        warehouse_ids = warehouse_coords[Axis.id_coordinates].values

        existing_ids = set(data.coords[Axis.id_coordinates].values) & set(warehouse_ids)
        new_ids = set(data.coords[Axis.id_coordinates].values) - set(warehouse_ids)

        if existing_ids and new_ids:  # detect returned the original
            raise NotImplementedError(
                "There should not be a case of both existing trace update and new components detection in update"
            )
        elif existing_ids:  # new frame trace update
            self._append(data, append_dim=Axis.frames_axis)

        elif new_ids:  # detect only returned new elements
            n_frames_to_backfill = len(warehouse_coords[Axis.frames_axis]) - len(
                data.coords[Axis.frames_axis]
            )

            if n_frames_to_backfill > 0:
                # grab coordinates in warehouse
                warehouse_frames = warehouse_coords[Axis.frame_coordinates].values[
                    :n_frames_to_backfill
                ]
                warehouse_times = warehouse_coords[Axis.time_coordinates].values[
                    :n_frames_to_backfill
                ]

                # Create zeros array with same shape as data but for missing frames
                zeros = xr.DataArray(
                    np.zeros((data.sizes[Axis.component_axis], n_frames_to_backfill)),
                    dims=(Axis.component_axis, Axis.frames_axis),
                    coords={
                        Axis.frame_coordinates: (Axis.frames_axis, warehouse_frames),
                        Axis.time_coordinates: (Axis.frames_axis, warehouse_times),
                    },
                )
                # Combine zeros and data along frames axis
                backfilled_data = xr.concat([zeros, data], dim=Axis.frames_axis)
            else:
                backfilled_data = data

            self._append(backfilled_data, append_dim=Axis.component_axis)


Traces = Annotated[xr.DataArray, TraceStore]
