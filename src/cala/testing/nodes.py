from collections.abc import Generator
from typing import Annotated as A

import numpy as np
from noob import Name

from cala.assets import Frame
from cala.testing.toy import FrameDims, Position, Toy


def single_cell_source(
    n_frames: int = 30,
    frame_dims: dict = None,
    cell_radii: int = 30,
    positions: list[dict] = None,
) -> Generator[A[Frame, Name("frame")]]:
    frame_dims = FrameDims(width=512, height=512) if frame_dims is None else FrameDims(**frame_dims)
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


def two_cells_source(
    n_frames: int = 30,
    frame_dims: dict = None,
    cell_radii: int = 30,
    positions: list[dict] = None,
) -> Generator[A[Frame, Name("frame")]]:
    frame_dims = FrameDims(width=512, height=512) if frame_dims is None else FrameDims(**frame_dims)
    traces = [np.array(range(0, n_frames)), np.array([0, *range(30 - 1, 0, -1)])]
    if positions is None:
        positions = [Position(width=206, height=206), Position(width=306, height=306)]
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


def two_overlapping_source(
    n_frames: int = 30,
    frame_dims: dict = None,
    cell_radii: int = 30,
    positions: list[dict] = None,
) -> Generator[A[Frame, Name("frame")]]:
    frame_dims = FrameDims(width=512, height=512) if frame_dims is None else FrameDims(**frame_dims)
    traces = [np.array(range(0, n_frames)), np.array([0, *range(n_frames - 1, 0, -1)])]
    if positions is None:
        positions = [Position(width=236, height=236), Position(width=276, height=276)]
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
