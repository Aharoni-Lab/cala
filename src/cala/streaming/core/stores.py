from enum import Enum

from xarray import DataArray


class Observable(DataArray):
    """Base class for observable objects."""

    __slots__ = ()


class Footprints(Observable):
    __slots__ = ()


class Traces(Observable):
    __slots__ = ()


class Component(Enum):
    NEURON = "neuron"
    BACKGROUND = "background"


class ComponentTypes(list):
    def __init__(self, iterable=None):
        super(ComponentTypes, self).__init__()
        if iterable:
            self.extend(iterable)

    def _check_element(self, item):
        if not isinstance(item, Component):
            raise ValueError("Item must be an Component")
        return item

    def append(self, item):
        super(ComponentTypes, self).append(self._check_element(item))

    def insert(self, index, item):
        super(ComponentTypes, self).insert(index, self._check_element(item))

    def extend(self, iterable):
        for item in iterable:
            self.append(item)

    def __add__(self, other):
        result = ComponentTypes(self)
        result.extend(other)
        return result

    def __iadd__(self, other):
        self.extend(other)
        return self

    def __setitem__(self, index, item):
        if isinstance(index, slice):
            items = []
            for i in item:
                items.append(self._check_element(i))
            super(ComponentTypes, self).__setitem__(index, items)
        else:
            super(ComponentTypes, self).__setitem__(index, self._check_element(item))
