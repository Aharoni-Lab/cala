from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, DefaultDict, Optional, Set
from uuid import uuid4

from cala.streaming.types import ComponentType


@dataclass
class Registry:
    """Manages the registration and lookup of fluorescent components."""

    type_to_ids: DefaultDict[ComponentType, Set[str]] = field(
        default_factory=lambda: defaultdict(set)
    )
    id_to_type: Dict[str, ComponentType] = field(default_factory=dict)

    @property
    def ids(self) -> Set[str]:
        """Returns all component IDs."""
        return set(self.id_to_type.keys())

    @property
    def n_components(self) -> int:
        """Returns the number of components."""
        return len(self.id_to_type)

    def create(self, component_type: ComponentType | str) -> str:
        if isinstance(component_type, str):
            component_type = ComponentType(component_type)
        hex_id = uuid4().hex
        self.type_to_ids[component_type].add(hex_id)
        self.id_to_type[hex_id] = component_type
        return hex_id

    def remove(self, component_id: str) -> None:
        """Remove a component by its ID."""
        component_type = self.id_to_type.pop(component_id)
        self.type_to_ids[component_type].remove(component_id)

    def get_type_by_id(self, component_id: str) -> Optional[ComponentType]:
        """Get a component by its ID."""
        return self.id_to_type.get(component_id)

    def get_id_by_type(self, component_type: ComponentType | str) -> Set[str]:
        """Get all component IDs of a specific type.

        Args:
            component_type: The type name as a string (e.g. "neuron", "background")
        """
        return self.type_to_ids.get(component_type)

    def clear(self) -> None:
        self.type_to_ids.clear()
        self.id_to_type.clear()

    def create_many(self, count: int, component_type: ComponentType | str) -> Set[str]:
        return set(self.create(component_type) for _ in range(count))
