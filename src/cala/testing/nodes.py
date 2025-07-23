from typing import Annotated as A

from noob import Name
from noob.node import Node


class NodeA(Node):
    a: int
    b: str = "C"

    def process(self, up: float, down: list[str]) -> A[float, Name("hoy")]:
        return self.a * up


class NodeB(Node):
    c: int = 3
    d: str

    def process(self, left: float, right: float = 1.2) -> dict[str, float]:
        return {self.d: left**right}
