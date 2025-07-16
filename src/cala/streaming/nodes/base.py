from abc import ABC, abstractmethod


class Node(ABC):
    id_: str

    @abstractmethod
    def process(self, *args, **kwargs): ...
