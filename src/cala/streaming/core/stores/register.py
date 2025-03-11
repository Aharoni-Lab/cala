from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, DefaultDict, List, Type

from cala.streaming.types import FluorescentObject


@dataclass
class Register:
    """Manages the registration and lookup of fluorescent components."""

    type_to_ids: DefaultDict[Type["FluorescentObject"], List[str]] = field(
        default_factory=lambda: defaultdict(list)
    )
    id_to_type: Dict[str, Type["FluorescentObject"]] = field(default_factory=dict)

    @property
    def ids(self) -> List[str]:
        """Returns all component IDs."""
        return list(self.id_to_type.keys())

    @property
    def types(self) -> List[Type["FluorescentObject"]]:
        """Returns all component types."""
        return list(self.type_to_ids.keys())

    @property
    def n_components(self) -> int:
        """Returns the number of components."""
        return len(self.id_to_type)

    def create(self, id_: str, type_: Type["FluorescentObject"]) -> None:
        """Create a new component."""
        if not issubclass(type_, FluorescentObject):
            raise TypeError(
                f"Component type {type_} must inherit from FluorescentObject"
            )
        self.type_to_ids[type_].append(id_)
        self.id_to_type[id_] = type_

    def remove(self, id_: str) -> None:
        """Remove a component by its ID."""
        type_ = self.id_to_type.pop(id_)
        self.type_to_ids[type_].remove(id_)

    def get_type_by_id(self, id_: str) -> Type["FluorescentObject"]:
        """Get a component by its ID."""
        return self.id_to_type[id_]

    def get_id_by_type(self, type_: Type["FluorescentObject"]) -> List[str]:
        """Get all component IDs of a specific type.

        Args:
            type_: The type name as a string (e.g. "neuron", "background")
        """
        return self.type_to_ids.get(type_, [])

    def clear(self) -> None:
        self.type_to_ids.clear()
        self.id_to_type.clear()

    def create_many(
            self, ids: List[str], types: List[Type["FluorescentObject"]]
    ) -> None:
        """Create multiple components."""
        for id_, type_ in zip(ids, types):
            self.create(id_, type_)
