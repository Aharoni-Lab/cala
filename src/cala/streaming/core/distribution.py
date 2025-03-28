from dataclasses import dataclass
from typing import Type, Optional, get_origin, Annotated

import xarray as xr

from cala.streaming.core import ObservableStore


@dataclass
class Distributor:
    """Manages a collection of fluorescent components (neurons and background).

    This class serves as a central manager for different storages,
    including spatial footprints, temporal traces, and various statistics.
    """

    _: int = 0

    def get(self, type_: Type) -> Optional[ObservableStore]:
        """Retrieve a specific Observable instance based on its type.

        Args:
            type_ (Type): The type of Observable to retrieve (e.g., Footprints, Traces).

        Returns:
            Optional[ObservableStore]: The requested Observable instance if found, None otherwise.
        """
        store_type = self._get_store_type(type_)
        if store_type is None:
            return
        for attr_name, attr_type in self.__annotations__.items():
            if attr_type == store_type:
                return getattr(self, attr_name).warehouse

    def init(self, result: xr.DataArray, type_: Type) -> None:
        """Store a DataArray results in their appropriate Observable containers.

        This method automatically determines the correct storage location based on the
        type of the input DataArray.

        Args:
            result: A single xr.DataArray to be stored. Must correspond to a valid Observable type.
            type_: type of the result. If an observable, should be an Annotated type that links to Store class.
        """
        target_store_type = self._get_store_type(type_)
        if target_store_type is None:
            return

        store_name = target_store_type.__name__.lower()
        # Add to annotations
        self.__annotations__[store_name] = target_store_type
        # Create and set the store
        setattr(self, store_name, target_store_type(result))

    def update(
            self,
            result: xr.DataArray | tuple[xr.DataArray, ...],
            type_: Type | tuple[Type, ...],
    ) -> None:
        """Update appropriate Observable containers with result DataArray(s).

        This method automatically determines the correct storage location based on the
        type of the input DataArray(s).

        Args:
            result: A single xr.DataArray or tuple of DataArrays to be stored. Must correspond to valid Observable types.
            type_: Type or tuple of types of the result(s). If an observable, should be an Annotated type that links to Store class.
        """
        # Convert single inputs to tuples for uniform handling
        results = (result,) if isinstance(result, xr.DataArray) else result
        types = (type_,) if isinstance(type_, type) else type_

        if len(results) != len(types):
            raise ValueError("Number of results must match number of types")

        for r, t in zip(results, types):
            target_store_type = self._get_store_type(t)
            if target_store_type is None:
                continue

            store_name = target_store_type.__name__.lower()
            getattr(self, store_name).update(r)

    @staticmethod
    def _get_store_type(type_: Type) -> type | None:
        if get_origin(type_) is Annotated:
            if issubclass(type_.__metadata__[0], ObservableStore):
                return type_.__metadata__[0]
