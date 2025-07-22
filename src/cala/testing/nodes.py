from cala.models import Node
from cala.models.params import Parameters


class NodeAParams(Parameters):
    a: int
    b: str = "B"

    def validate(self): ...


class NodeA(Node):
    def process(self, m: float, n: list[str]) -> dict[str, float]:
        return {n[0]: m}
