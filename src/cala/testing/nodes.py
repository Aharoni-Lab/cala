from collections.abc import Generator
from typing import Annotated as A

import numpy as np
from noob import Name
from noob.node import Node

from cala.assets import Frame
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
    n_frames: int = 30,
    frame_dims: dict = None,
    cell_radii: int = 30,
    positions: list[dict] = None,
    traces: list[np.ndarray] = None,
) -> Generator[A[Frame, Name("frame")]]:
    frame_dims = FrameDims(width=512, height=512) if frame_dims is None else FrameDims(**frame_dims)
    if traces is None:
        traces = [np.array(range(0, n_frames))]
    if positions is None:
        positions = [Position(width=256, height=256)]
    else:
        positions = [Position(**position) for position in positions]

    toy = Toy(
        n_frames=n_frames,
        frame_dims=frame_dims,
        cell_radii=cell_radii,
        cell_positions=positions,
        cell_traces=traces,
    )
    return toy.movie_gen()
