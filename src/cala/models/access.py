import xarray as xr
from xarray_validate import DataArraySchema


@xr.register_dataarray_accessor("validate")
class DaValidator:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def against_schema(self, schema: DataArraySchema) -> bool:
        """
        Validates the DataArray against a given xarray-schema model.
        Raises ValueError if validation fails.
        """

        schema.validate(self._obj)

        return True


@xr.register_dataarray_accessor("volumize")
class DaVolumizer:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def dim_with_coords(self, dim: str, coords: list[str]) -> xr.DataArray:
        return self._obj.expand_dims(dim).assign_coords(
            {c: (dim, [self._obj[c].item()]) for c in coords}
        )
