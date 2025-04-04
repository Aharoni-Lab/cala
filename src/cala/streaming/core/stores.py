from dataclasses import dataclass

import xarray as xr


@dataclass
class ObservableStore:
    """Base class for observable objects in calcium imaging data."""

    _warehouse: xr.DataArray

    @property
    def warehouse(self) -> xr.DataArray:
        return self._warehouse

    @warehouse.setter
    def warehouse(self, value: xr.DataArray) -> None:
        self._warehouse = value
