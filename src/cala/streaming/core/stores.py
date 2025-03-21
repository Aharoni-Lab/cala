from dataclasses import dataclass

import xarray as xr


@dataclass
class ObservableStore:
    """Base class for observable objects in calcium imaging data."""

    _warehouse: xr.DataArray
