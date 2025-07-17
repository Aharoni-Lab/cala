from abc import ABC, abstractmethod
from typing import Any


class Node(ABC):
    id_: str

    @abstractmethod
    def process(self, *args: Any, **kwargs: Any) -> Any: ...
