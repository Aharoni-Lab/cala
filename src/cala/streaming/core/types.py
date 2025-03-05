import xarray as xr


# There's a footprint class. Then there's a footprint (DataArray) type.
# There's a footprints class, then there's a footprints (DataArray) type.
# The actual data exist as groups of DataArrays, in observableS classes.
# the solo observable class has a pointer to a named index of the group.
# There's a key-value store that links fluorescing objects to an index of the arrays. This exists in a manager class.
# Manager does all the retrieval / addition / deletion / update stuff.
# The actual implementations are done in


class Footprint(xr.DataArray):
    pass


class Trace(xr.DataArray):
    pass


class Footprints(xr.DataArray):
    pass


class Traces(xr.DataArray):
    pass

# @dataclass
# class FluorescingObject:
#     id: str
#
#
# @dataclass
# class Neuron(FluorescingObject):
#     pass
#
# @dataclass
# class Background(FluorescingObject):
#     pass


# ----------------------------------- #


# class NeuronFootprint(Neuron, Footprint):
#     pass
#
#
# class BackgroundFootprint(Background, Footprint):
#     pass
#
#
# NeuronFootprints: TypeAlias = list[NeuronFootprint]
# BackgroundFootprints: TypeAlias = list[BackgroundFootprint]
#
#
# class NeuronTrace(Trace, Neuron):
#     pass
#
#
# class BackgroundTrace(Trace, Background):
#     pass
#
#
# NeuronTraces: TypeAlias = list[NeuronTrace]
# BackgroundTraces: TypeAlias = list[BackgroundTrace]
