from typing import Annotated

from xarray import DataArray


class ObservableStore(DataArray):
    """Base class for observable objects in calcium imaging data.

    Extends xarray.DataArray.
    """

    __slots__ = ()


class FootprintStore(ObservableStore):
    """Spatial footprints of identified components in calcium imaging data.

    Represents the spatial distribution patterns of components (neurons or background)
    in the field of view. Each footprint typically contains the spatial extent and
    intensity weights of a component.
    """

    __slots__ = ()


Footprints = Annotated[DataArray, ObservableStore(FootprintStore)]


class TraceStore(ObservableStore):
    """Temporal activity traces of identified components.

    Contains the time-varying fluorescence signals of components across frames,
    representing their activity patterns over time.
    """

    __slots__ = ()


Traces = Annotated[DataArray, ObservableStore(TraceStore)]
