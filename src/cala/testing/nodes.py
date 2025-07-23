from collections.abc import Generator
from typing import Annotated as A

import numpy as np
from noob import Name
from noob.node import Node

from cala.models import Frame
from cala.testing.toy import FrameDims, Position, Toy


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


class SingleCell(Node):
    frame_dims: FrameDims = FrameDims(width=512, height=512)
    positions: list[Position] = [Position(width=256, height=256)]
    traces: list[np.ndarray] = [np.array(range(0, 500))]

    def __post_init__(self) -> None:
        self.toy = Toy(
            n_frames=500,
            frame_dims=self.frame_dims,
            cell_radii=30,
            cell_positions=self.positions,
            cell_traces=self.traces,
            confidences=[],
        )

    def process(self) -> A[Generator[Frame], Name("frame")]:
        return self.toy.iter_frame()
