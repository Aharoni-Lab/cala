from typing import Annotated

import xarray as xr

from cala.streaming.core import ObservableStore, Axis


class FootprintStore(ObservableStore):
    """Spatial footprints of identified components.

    Represents the spatial distribution patterns of components (neurons or background)
    in the field of view. Each footprint typically contains the spatial extent and
    intensity weights of a component.
    """

    def update(self, data: xr.DataArray):
        if len(data) == 0:
            return

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


Footprints = Annotated[xr.DataArray, FootprintStore]


class TraceStore(ObservableStore):
    """Temporal activity traces of identified components.

    Contains the time-varying fluorescence signals of components across frames,
    representing their activity patterns over time.
    """

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
        elif existing_ids:  # new frame trace update
            self.warehouse = xr.concat(
                [self.warehouse, data],
                dim=Axis.frames_axis,
            )
        elif new_ids:  # detect only returned new elements
            n_frames_to_backfill = len(self.warehouse.coords[Axis.frames_axis]) - len(
                data.coords[Axis.frames_axis]
            )

            if n_frames_to_backfill > 0:
                # Create zeros array with same shape as data but for missing frames
                zeros = xr.zeros_like(data).expand_dims(
                    Axis.frames_axis, n_frames_to_backfill
                )
                # Combine zeros and data along frames axis
                backfilled_data = xr.concat([zeros, data], dim=Axis.frames_axis)
            else:
                backfilled_data = data

            self.warehouse = xr.concat(
                [self.warehouse, backfilled_data],
                dim=Axis.component_axis,
            )


Traces = Annotated[xr.DataArray, TraceStore]
