from dataclasses import dataclass, field

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

    def _generate_movie_template(self):
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

    def _format_traces(self) -> xr.DataArray:
        traces = []
        for trace, id_ in zip(self.cell_traces, self.cell_ids):
            traces.append(
                xr.DataArray(
                    trace,
                    dims=Axis.frames_axis,
                )
                .expand_dims(Axis.component_axis)
                .assign_coords({Axis.id_coordinates: (Axis.component_axis, [id_])})
            )

        return xr.concat(traces, dim=Axis.component_axis)

    def _generate_movie(self, footprints, traces) -> xr.DataArray:
        movie = self._generate_movie_template()
        movie += (footprints @ traces).transpose(Axis.frames_axis, *Axis.spatial_axes)
        return movie

    def run(self):
        self.footprints_ = self._build_footprints()
        self.traces_ = self._format_traces()
        movie = self._generate_movie(self.footprints_, self.traces_)
        return movie


def main():
    frame_dims = FrameSize(width=1024, height=512)
    cell_positions = [Position(width=200, height=300)]
    cell_traces = [np.array(range(100))]

    movie_generator = Simulator(
        n_frames=100,
        frame_dims=frame_dims,
        cell_radii=20,
        cell_positions=cell_positions,
        cell_traces=cell_traces,
    )

    sample_movie = movie_generator.run()

    Plotter(output_dir="../..").write_movie(sample_movie)


if __name__ == "__main__":
    main()
