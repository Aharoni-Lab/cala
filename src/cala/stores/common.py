import logging
from typing import Annotated

import numpy as np
import xarray as xr

from cala.models.axis import AXIS
from cala.models.store import Store

logger = logging.getLogger(__name__)


class FootprintStore(Store):
    """Spatial footprints of identified components.

    Represents the spatial distribution patterns of components (neurons or background)
    in the field of view. Each footprint typically contains the spatial extent and
    intensity weights of a component.
    """

    def update(self, data: xr.DataArray) -> None:
        if len(data) == 0:
            return None

        existing_ids = set(data.coords[AXIS.id_coord].values) & set(
            self.warehouse.coords[AXIS.id_coord].values
        )
        new_ids = set(data.coords[AXIS.id_coord].values) - set(
            self.warehouse.coords[AXIS.id_coord].values
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
                dim=AXIS.component_dim,
            )
        return None


Footprints = Annotated[xr.DataArray, FootprintStore]


class TraceStore(Store):
    """Temporal activity traces of identified components.

    Contains the time-varying fluorescence signals of components across frames,
    representing their activity patterns over time.
    """

    persistent = True

    @property
    def warehouse(self) -> xr.DataArray:
        return (
            xr.open_zarr(self.store_path)
            .isel({AXIS.frames_dim: slice(-self.peek_size, None)})
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

        warehouse_ids = warehouse_coords[AXIS.id_coord].values

        existing_ids = set(data.coords[AXIS.id_coord].values) & set(warehouse_ids)
        new_ids = set(data.coords[AXIS.id_coord].values) - set(warehouse_ids)

        if existing_ids and new_ids:  # detect returned the original
            raise NotImplementedError(
                "There should not be a case of both existing trace update and new components detection in update"
            )
        elif existing_ids:  # new frame trace update
            self._append(data, append_dim=AXIS.frames_dim)

        elif new_ids:  # detect only returned new elements
            n_frames_to_backfill = len(warehouse_coords[AXIS.frames_dim]) - len(
                data.coords[AXIS.frames_dim]
            )

            if n_frames_to_backfill > 0:
                # grab coordinates in warehouse
                warehouse_frames = warehouse_coords[AXIS.frame_coord].values[:n_frames_to_backfill]
                warehouse_times = warehouse_coords[AXIS.timestamp_coord].values[
                    :n_frames_to_backfill
                ]

                # Create zeros array with same shape as data but for missing frames
                zeros = xr.DataArray(
                    np.zeros((data.sizes[AXIS.component_dim], n_frames_to_backfill)),
                    dims=(AXIS.component_dim, AXIS.frames_dim),
                    coords={
                        AXIS.frame_coord: (AXIS.frames_dim, warehouse_frames),
                        AXIS.timestamp_coord: (AXIS.frames_dim, warehouse_times),
                    },
                )
                # Combine zeros and data along frames axis
                backfilled_data = xr.concat([zeros, data], dim=AXIS.frames_dim)
            else:
                backfilled_data = data

            self._append(backfilled_data, append_dim=AXIS.component_dim)


Traces = Annotated[xr.DataArray, TraceStore]
