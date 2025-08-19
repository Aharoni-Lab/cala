from collections.abc import Generator
from typing import Annotated as A

import numpy as np
from noob import Name, process_method
from pydantic import BaseModel, PrivateAttr, model_validator

from cala.assets import Frame
from cala.testing.toy import FrameDims, Position, Toy


class MovieSource(BaseModel):
    n_frames: int = 50
    frame_dims: FrameDims | dict[str, int] | None = None
    cell_radii: int = 30
    positions: list[dict | Position] | None = None
    _toy: Toy = PrivateAttr(None)
    _traces: list[np.ndarray] = PrivateAttr(None)

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


class SingleCellSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self):
        self.frame_dims = (
            FrameDims(width=512, height=512)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )
        self._traces = [np.array(range(0, self.n_frames))]

        if self.positions is None:
            self.positions = [Position(width=256, height=256)]
        else:
            self.positions = [Position(**position) for position in self.positions]

        self._toy = self._build_toy()
        return self


class TwoCellsSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self):
        self.frame_dims = (
            FrameDims(width=512, height=512)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )

        self._traces = [
            np.array(range(0, self.n_frames)),
            np.array([0, *range(self.n_frames - 1, 0, -1)]),
        ]
        if self.positions is None:
            self.positions = [Position(width=206, height=206), Position(width=306, height=306)]
        else:
            self.positions = [Position(**position) for position in self.positions]

        self._toy = self._build_toy()
        return self


class TwoOverlappingSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self):
        self.frame_dims = (
            FrameDims(width=512, height=512)
            if self.frame_dims is None
            else FrameDims(**self.frame_dims)
        )
        self._traces = [
            np.array(range(0, self.n_frames)),
            np.array([0, *range(self.n_frames - 1, 0, -1)]),
        ]

        if self.positions is None:
            self.positions = [Position(width=236, height=236), Position(width=276, height=276)]
        else:
            self.positions = [Position(**position) for position in self.positions]

        self._toy = self._build_toy()
        return self


class SeparateSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self):
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
            np.zeros(self.n_frames, dtype=float),
            np.ones(self.n_frames, dtype=float),
            np.array(range(self.n_frames), dtype=float),
            np.array([0, *range(self.n_frames - 1, 0, -1)], dtype=float),
        ]

        self._toy = self._build_toy()
        return self


class ConnectedSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self):
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
            np.zeros(self.n_frames, dtype=float),
            np.ones(self.n_frames, dtype=float),
            np.array(range(self.n_frames), dtype=float),
            np.array([0, *range(self.n_frames - 1, 0, -1)], dtype=float),
        ]

        self._toy = self._build_toy()
        return self


class GradualOnSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self):
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
        decr = np.array(range(self.n_frames - 1, 0, -1), dtype=float)
        sine = np.abs(np.sin(np.linspace(0, 2 * np.pi, self.n_frames - gap)) * self.n_frames)
        incr = np.array(range(self.n_frames - gap * 2), dtype=float)
        rand = np.random.random(self.n_frames - gap * 3) * self.n_frames
        const = np.ones(self.n_frames - gap * 4, dtype=float) * self.n_frames

        self._traces = [
            np.pad(decr, (1, 0), mode="constant", constant_values=0),
            np.pad(sine, (gap, 0), mode="constant", constant_values=0),
            np.pad(incr, (gap * 2, 0), mode="constant", constant_values=0),
            np.pad(rand, (gap * 3, 0), mode="constant", constant_values=0),
            np.pad(const, (gap * 4, 0), mode="constant", constant_values=0),
        ]

        self._toy = self._build_toy()
        return self


class SplitOffSource(MovieSource):
    @model_validator(mode="after")
    def complete_model(self): ...
