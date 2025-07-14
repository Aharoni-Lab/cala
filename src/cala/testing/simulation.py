from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import xarray as xr
from pydantic import BaseModel
from skimage.morphology import disk

from cala.gui.plots import Plotter
from cala.streaming.core import Axis


class FrameSize(BaseModel):
    width: int
    height: int


class Position(BaseModel):
    width: int
    height: int


@dataclass
class Simulator:
    n_frames: int
    frame_dims: FrameSize
    cell_radii: int | list[int]
    cell_positions: list[Position]
    cell_traces: list[np.ndarray]
    cell_ids: list[str] | None = None

    footprints_: xr.DataArray = field(init=False)
    traces_: xr.DataArray = field(init=False)

    """If none, auto populated as cell_{idx}."""

    def __post_init__(self):
        assert self.n_frames > 0

        self.cell_radii = (
            [self.cell_radii] * len(self.cell_positions)
            if isinstance(self.cell_radii, int)
            else self.cell_radii
        )
        for position, radius in zip(self.cell_positions, self.cell_radii):
            assert 0 < np.min([position.width, position.height]) - radius
            assert position.width + radius < self.frame_dims.width
            assert position.height + radius < self.frame_dims.height

        assert len(self.cell_positions) == len(self.cell_traces)

        for cell_trace in self.cell_traces:
            assert self.n_frames == len(cell_trace)

        if self.cell_ids is None:
            self.cell_ids = [f"cell_{idx}" for idx, _ in enumerate(self.cell_positions)]
        assert len(self.cell_ids) == len(self.cell_traces)

        self.footprints_ = self._build_footprints()
        self.traces_ = self._build_traces()

    def _build_movie_template(self):
        return xr.DataArray(
            np.zeros((self.n_frames, self.frame_dims.height, self.frame_dims.width)),
            dims=[Axis.frames_axis, *Axis.spatial_axes],
        ).astype(np.float32)

    def _generate_footprint(self, radius: int, position: Position, id_: str) -> xr.DataArray:
        footprint = xr.DataArray(
            np.zeros((self.frame_dims.height, self.frame_dims.width)),
            dims=Axis.spatial_axes,
        )

        shape = disk(radius).astype(np.float32)

        width_slice = slice(position.width - radius, position.width + radius + 1)
        height_slice = slice(position.height - radius, position.height + radius + 1)

        footprint.loc[{"height": height_slice, "width": width_slice}] = shape

        return footprint.expand_dims(Axis.component_axis).assign_coords(
            {Axis.id_coordinates: (Axis.component_axis, [id_])}
        )

    def _build_footprints(self) -> xr.DataArray:
        footprints = []
        for radius, position, id_ in zip(self.cell_radii, self.cell_positions, self.cell_ids):
            footprints.append(self._generate_footprint(radius, position, id_))

        return xr.concat(footprints, dim=Axis.component_axis)

    def _format_trace(self, trace: np.ndarray, id_: str) -> xr.DataArray:
        return (
            xr.DataArray(
                trace,
                dims=Axis.frames_axis,
            )
            .expand_dims(Axis.component_axis)
            .assign_coords(
                {
                    Axis.id_coordinates: (Axis.component_axis, [id_]),
                    Axis.frames_axis: range(trace.size),
                }
            )
        )

    def _build_traces(self) -> xr.DataArray:
        traces = []
        for trace, id_ in zip(self.cell_traces, self.cell_ids):
            traces.append(self._format_trace(trace, id_))

        return xr.concat(traces, dim=Axis.component_axis)

    def _build_movie(self, footprints, traces) -> xr.DataArray:
        movie = self._build_movie_template()
        movie += (footprints @ traces).transpose(Axis.frames_axis, *Axis.spatial_axes)
        return movie

    def make_movie(self):
        movie = self._build_movie(self.footprints_, self.traces_)
        return movie

    def add_cell(self, position: Position, radius: int, trace: np.ndarray, id_: str):
        new_footprint = self._generate_footprint(radius, position, id_)
        self.footprints_ = xr.concat([self.footprints_, new_footprint], dim=Axis.component_axis)

        new_trace = self._format_trace(trace, id_)
        self.traces_ = xr.concat([self.traces_, new_trace], dim=Axis.component_axis)

    def drop_cell(self, id_: str | Iterable[str]):
        id_ = {id_} if isinstance(id_, str) else set(id_)

        id_coords = set(self.footprints_.coords["id_"].values.tolist())

        keep_ids = list(id_coords - id_)

        self.footprints_ = (
            self.footprints_.set_xindex(Axis.id_coordinates)
            .sel({Axis.id_coordinates: keep_ids})
            .reset_index(Axis.id_coordinates)
        )
        self.traces_ = (
            self.traces_.set_xindex(Axis.id_coordinates)
            .sel({Axis.id_coordinates: keep_ids})
            .reset_index(Axis.id_coordinates)
        )

    @property
    def footprints(self) -> xr.DataArray:
        if self.footprints_:
            return self.footprints_
        else:
            raise ValueError("No footprints available")

    @property
    def traces(self) -> xr.DataArray:
        if self.traces_:
            return self.traces_
        else:
            raise ValueError("No traces available")


def main():
    frame_dims = FrameSize(width=1024, height=512)
    cell_positions = [Position(width=200, height=300)]
    cell_traces = [np.array(range(100))]

    simulator = Simulator(
        n_frames=100,
        frame_dims=frame_dims,
        cell_radii=20,
        cell_positions=cell_positions,
        cell_traces=cell_traces,
    )

    cell_position = Position(width=400, height=300)
    cell_trace = np.array(range(100, 0, -1))

    simulator.add_cell(cell_position, 10, cell_trace, "cell_a")

    simulator.drop_cell("cell_a")

    sample_movie = simulator.make_movie()

    Plotter(output_dir="../..").write_movie(sample_movie)


if __name__ == "__main__":
    main()
