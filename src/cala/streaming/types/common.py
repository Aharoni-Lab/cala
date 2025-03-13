from typing import Type, Any  # NewType, Protocol

from xarray import DataArray


class Observable(DataArray):
    """Base class for observable objects."""

    __slots__ = ()


class Footprints(Observable):
    __slots__ = ()


class Traces(Observable):
    __slots__ = ()


class FluorescentObject:
    """Base type for any fluorescent object detected."""

    pass


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


_generated_classes = generate_cross_product_classes()
NeuronFootprints = _generated_classes["NeuronFootprints"]
NeuronTraces = _generated_classes["NeuronTraces"]
BackgroundFootprints = _generated_classes["BackgroundFootprints"]
BackgroundTraces = _generated_classes["BackgroundTraces"]


def find_intersection_type_of(base_type: Type, instance: Any) -> Type:
    component_types = set(base_type.__subclasses__())
    try:
        parent_types = set(instance.__bases__)
    except AttributeError:
        try:
            return {
                component_type
                for component_type in component_types
                if isinstance(instance, component_type)
            }.pop()
        except KeyError:
            raise TypeError(
                f"The instance type {instance.__class__.__name__} does not inherit from one of the component types: {component_types}"
            )

    component_type = parent_types & component_types
    if len(component_type) != 1:
        raise TypeError(
            f"The instance type {instance.__class__.__name__} does not inherit from one of the component types: {component_types}"
        )
    return component_type.pop()
