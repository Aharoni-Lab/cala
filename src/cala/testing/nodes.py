from cala.models import Node
from cala.models.params import Params


class NodeAParams(Params):
    a: int
    b: str = "B"

    def validate(self): ...


class NodeA(Node):
    def process(self, m: float, n: list[str]) -> dict[str, float]:
        return {n[0]: m}
