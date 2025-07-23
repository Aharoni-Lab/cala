from typing import Annotated as A, Generator

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


def single_cell_source(
    n_frames: int = 500,
    frame_dims: FrameDims = FrameDims(width=512, height=512),
    cell_radii: int = 30,
    positions: list[Position] = None,
    traces: list[np.ndarray] = None,
) -> Generator[A[Frame, Name("frame")]]:
    if traces is None:
        traces = [np.array(range(0, 500))]
    if positions is None:
        positions = [Position(width=256, height=256)]

    toy = Toy(
        n_frames=n_frames,
        frame_dims=frame_dims,
        cell_radii=cell_radii,
        cell_positions=positions,
        cell_traces=traces,
        confidences=[],
    )
    return toy.movie_gen()
