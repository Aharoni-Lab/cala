import xarray as xr

from xarray_validate import DataArraySchema, DimsSchema, CoordsSchema, DTypeSchema

from cala.models.dim import Coord
from cala.models.entity import Entity


@xr.register_dataarray_accessor("validate")
class DaValidator:
    def __init__(self, xarray_obj: xr.DataArray):
        self._obj = xarray_obj

    def _build_coord_schema(self, coords: list[Coord]) -> CoordsSchema:
        return CoordsSchema(
            {
                c.name: DataArraySchema(dims=DimsSchema([c.dim]), dtype=DTypeSchema(c.dtype))
                for c in coords
            }
        )

    def _build_entity_schema(self, schema: Entity) -> DataArraySchema:
        coords_schema = self._build_coord_schema(schema.coords) if schema.coords else None

        return DataArraySchema(
            dims=DimsSchema(tuple(dim.name for dim in schema.dims)),
            coords=coords_schema,
            dtype=DTypeSchema(schema.dtype),
        )

    def against_schema(self, schema: Entity):
        """
        Validates the DataArray against a given Pydantic schema.
        Raises ValueError if validation fails.
        """
        da_schema = self._build_entity_schema(schema)

        da_schema.validate(self._obj)

        # check coordinate data rules
        # check actual value rules

        return True
