from cala.models import Node


class NodeA(Node):
    a: int
    b: str = "B"

    def process(self, m: float, n: list[str]) -> dict[str, float]:
        return {n[0]: m}
