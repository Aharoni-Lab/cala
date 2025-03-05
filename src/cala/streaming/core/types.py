from dataclasses import dataclass
from typing import TypeAlias, Tuple

import xarray as xr


@dataclass
class Footprint(xr.DataArray):
    id: str
    shape: Tuple[int, ...]


@dataclass
class Trace(xr.DataArray):
    id: str
    shape: Tuple[int, ...]


Footprints: TypeAlias = list[Footprint]
Traces: TypeAlias = list[Trace]


@dataclass
class FluorescingObject:
    id: str
    footprint: Footprint
    trace: Trace

    def __post_init__(self):
        if not all({self.footprint.id, self.trace.id, self.id}):
            raise ValueError("IDs in footprint and trace must be identical")


@dataclass
class Neuron(FluorescingObject):
    spike_train: xr.DataArray


@dataclass
class Background(FluorescingObject):
    pass


# ---------------------------------- #


@dataclass
class NeuronFootprint(Neuron, Footprint):
    pass


@dataclass
class BackgroundFootprint(Background, Footprint):
    pass


NeuronFootprints: TypeAlias = list[NeuronFootprint]
BackgroundFootprints: TypeAlias = list[BackgroundFootprint]


@dataclass
class NeuronTrace(Trace, Neuron):
    pass


@dataclass
class BackgroundTrace(Trace, Background):
    pass


NeuronTraces: TypeAlias = list[NeuronTrace]
BackgroundTraces: TypeAlias = list[BackgroundTrace]
