from cala.models import Node


class NodeA(Node):
    a: int
    b: str = "C"

    def process(self, m: float, n: list[str]) -> dict[str, float]:
        return {n[0]: m}


class NodeB(Node):
    c: int
    d: str = "D"

    def process(self, o: float, p: float) -> float:
        return o**p
