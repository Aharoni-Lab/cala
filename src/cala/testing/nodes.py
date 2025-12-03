from collections.abc import Generator
from typing import Annotated as A
from typing import Literal, Self

import numpy as np
from noob import Name, process_method
from pydantic import BaseModel, PrivateAttr, model_validator

from cala.assets.assets import Frame
from cala.testing.toy import FrameDims, Position, Toy


class MovieSource(BaseModel):
    n_frames: int = 50
    frame_dims: FrameDims | dict[str, int] | None = None
    cell_radii: int = 30
    positions: list[dict | Position] | None = None
    _toy: Toy = PrivateAttr(None)
    _traces: list[np.ndarray] = PrivateAttr(None)

    @property
    def toy(self) -> Toy:
        return self._toy

    def _build_toy(self) -> Toy:
        return Toy(
            n_frames=self.n_frames,
            frame_dims=self.frame_dims,
            cell_radii=self.cell_radii,
            cell_positions=self.positions,
            cell_traces=self._traces,
        )

    @process_method
    def process(self) -> Generator[A[Frame, Name("frame")]]:
        yield from self._toy.movie_gen()

    def _simulate_trace(
        self, type_: Literal["incr", "decr", "expo", "sine", "tanh"], start: int = 0
    ) -> np.ndarray:
        if type_ == "incr":
            kernel = np.array(range(self.n_frames - start), dtype=float)
        elif type_ == "decr":
            kernel = np.array(range(self.n_frames - 1 - start, -1, -1), dtype=float)
        elif type_ == "expo":
            kernel = (
                np.linspace(0, np.exp(3), self.n_frames - start)
                * np.exp(-np.linspace(0, np.exp(2), self.n_frames - start))
                * self.n_frames
            )
        elif type_ == "sine":
            kernel = np.abs(
                np.sin(np.linspace(0, 2 * np.pi, self.n_frames - start)) * self.n_frames
            )
        elif type_ == "tanh":
            kernel = np.tanh(np.linspace(0, 5, self.n_frames - start)) * self.n_frames
        else:
            raise NotImplementedError(type_)

        return np.pad(kernel, (start, 0), mode="constant", constant_values=0)


class SingleCellSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self) -> Self:
        self.frame_dims = (
            FrameDims(width=512, height=512)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )
        self._traces = [self._simulate_trace("incr")]

        if self.positions is None:
            self.positions = [Position(width=256, height=256)]
        else:
            self.positions = [Position(**position) for position in self.positions]

        self._toy = self._build_toy()
        return self


class TwoCellsSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self) -> Self:
        self.frame_dims = (
            FrameDims(width=512, height=512)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )

        self._traces = [self._simulate_trace("incr"), self._simulate_trace("decr", 1)]
        if self.positions is None:
            self.positions = [Position(width=206, height=206), Position(width=306, height=306)]
        else:
            self.positions = [Position(**position) for position in self.positions]

        self._toy = self._build_toy()
        return self


class TwoOverlappingSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self) -> Self:
        self.frame_dims = (
            FrameDims(width=512, height=512)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )
        self._traces = [self._simulate_trace("incr"), self._simulate_trace("decr", 1)]

        if self.positions is None:
            self.positions = [Position(width=236, height=236), Position(width=276, height=276)]
        else:
            self.positions = [Position(**position) for position in self.positions]

        self._toy = self._build_toy()
        return self


class SeparateSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self) -> Self:
        self.cell_radii = 3
        self.frame_dims = (
            FrameDims(width=50, height=50)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )
        self.positions = [
            Position(width=15, height=15),
            Position(width=15, height=35),
            Position(width=25, height=25),
            Position(width=35, height=35),
        ]
        self._traces = [
            self._simulate_trace("sine"),
            self._simulate_trace("tanh"),
            self._simulate_trace("incr"),
            self._simulate_trace("decr", 1),
        ]

        self._toy = self._build_toy()
        return self


class ConnectedSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self) -> Self:
        self.cell_radii = 8
        self.frame_dims = (
            FrameDims(width=50, height=50)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )
        self.positions = [
            Position(width=15, height=15),
            Position(width=15, height=35),
            Position(width=25, height=25),
            Position(width=35, height=35),
        ]
        self._traces = [
            self._simulate_trace("expo"),
            self._simulate_trace("sine"),
            self._simulate_trace("incr"),
            self._simulate_trace("decr", 1),
        ]

        self._toy = self._build_toy()
        return self


class GradualOnSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self) -> Self:
        self.n_frames = 100
        self.cell_radii = 8
        self.frame_dims = (
            FrameDims(width=50, height=50)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )
        self.positions = [
            Position(width=15, height=15),
            Position(width=15, height=35),
            Position(width=25, height=25),
            Position(width=35, height=35),
            Position(width=35, height=15),
        ]
        gap = 20
        self._traces = [
            # decr start with at least one 0 to prevent
            # deglowing from completely erasing the component
            self._simulate_trace("decr", start=1),
            self._simulate_trace("sine", start=gap),
            self._simulate_trace("incr", start=gap * 2),
            self._simulate_trace("expo", start=gap * 3),
            self._simulate_trace("tanh", start=gap * 4),
        ]

        self._toy = self._build_toy()
        return self


class SplitOffSource(MovieSource):
    """
    1. Two overlapping cells go up together till 25.
    2. One dims to 0 while the other continues to get brighter until 50.
    3. Dim cell grows to 50 while bright cell dims to 0.
    """

    @model_validator(mode="after")
    def complete_model(self) -> Self:
        self.n_frames = 100
        self.cell_radii = 8
        self.frame_dims = (
            FrameDims(width=50, height=50)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )
        self.positions = [
            Position(width=20, height=20),
            Position(width=30, height=30),
        ]
        self._traces = [
            np.array(
                [0, *range(1, int(self.n_frames / 2)), *range(int(self.n_frames / 2), 0, -1)],
                dtype=float,
            ),
            np.array(
                [
                    *range(int(self.n_frames / 4)),
                    *range(int(self.n_frames / 4), 0, -1),
                    *range(int(self.n_frames / 2)),
                ],
                dtype=float,
            ),
        ]
        self._toy = self._build_toy()
        return self
