from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Type

from .categories import FluorescentObject


@dataclass
class Registry:
    """Manages the registration and lookup of fluorescent components."""

    _components: Dict[int, FluorescentObject] = field(default_factory=dict)
    """The components in the registry."""

    @property
    def component_ids(self) -> Set[int]:
        """Returns all component IDs."""
        return set(self._components.keys())

    @property
    def n_components(self) -> int:
        """Returns the number of components."""
        return len(self._components)

    def add(self, component: FluorescentObject) -> None:
        """Add a new component."""
        self._components[component.id] = component

    def remove(self, component_id: int) -> Optional[FluorescentObject]:
        """Remove a component by its ID."""
        return self._components.pop(component_id, None)

    def get(self, component_id: int) -> Optional[FluorescentObject]:
        """Get a component by its ID."""
        return self._components.get(component_id)

    def get_by_type(self, component_type: Type[FluorescentObject]) -> List[int]:
        """Get all component IDs of a specific type."""
        return [
            component.id
            for component in self._components.values()
            if isinstance(component, component_type)
        ]

    def clear(self) -> None:
        """Remove all components."""
        self._components.clear()
