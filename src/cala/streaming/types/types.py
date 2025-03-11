from enum import EnumMeta
from typing import Type

from xarray import DataArray


class BetterEnum(EnumMeta):
    def __getitem__(cls, name):
        try:
            return super().__getitem__(name)
        except KeyError as e:
            options = ", ".join([f"'{key}'" for key in cls._member_map_.keys()])
            msg = f"Please choose one of {options}. '{name}' provided."
            raise ValueError(msg) from None


class Observable(DataArray):
    """Base class for observable objects."""

    __slots__ = ()


class Footprints(Observable):
    __slots__ = ()


class Traces(Observable):
    __slots__ = ()


class FluorescentObject:
    """Base type for any fluorescent object detected."""

    def __hash__(self):
        return hash(self.__class__.__name__)


class Neuron(FluorescentObject):
    """Type representing a detected neuron."""

    pass


class Background(FluorescentObject):
    """Type representing background components."""

    pass


# ----------------------------------- #


def generate_cross_product_classes() -> dict[str, Type]:
    """Generate all combinations of Observable subclasses and FluorescentObject subclasses."""
    generated_classes = {}
    for observable_class in Observable.__subclasses__():
        for component_class in FluorescentObject.__subclasses__():
            class_name = f"{component_class.__name__}{observable_class.__name__}"
            generated_classes[class_name] = type(
                class_name, (component_class, observable_class), {"__slots__": ()}
            )
    return generated_classes


# Generate all the cross-product classes
_generated_classes = generate_cross_product_classes()
NeuronFootprints = _generated_classes["NeuronFootprints"]
NeuronTraces = _generated_classes["NeuronTraces"]
BackgroundFootprints = _generated_classes["BackgroundFootprints"]
BackgroundTraces = _generated_classes["BackgroundTraces"]
