from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from pydantic import BaseModel
from skimage.morphology import disk

from cala.models.axis import AXIS


class FrameSize(BaseModel):
    width: int
    height: int


class Position(BaseModel):
    width: int
    height: int


@dataclass
class Toy:
    """
    Ex:

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

    """

    n_frames: int
    frame_dims: FrameSize
    cell_radii: int | list[int]
    cell_positions: list[Position]
    cell_traces: list[np.ndarray]
    cell_ids: list[str] | None = None
    confidences: list[float] = field(default_factory=list)

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
            assert np.min([position.width, position.height]) - radius > 0
            assert position.width + radius < self.frame_dims.width
            assert position.height + radius < self.frame_dims.height

        assert len(self.cell_positions) == len(self.cell_traces)

        for cell_trace in self.cell_traces:
            assert self.n_frames == len(cell_trace)

        if self.cell_ids is None:
            self.cell_ids = [f"cell_{idx}" for idx, _ in enumerate(self.cell_positions)]
        assert len(self.cell_ids) == len(self.cell_traces)

        if not self.confidences:
            self.confidences = [0.0] * len(self.cell_ids)

        self.footprints_ = self._build_footprints()
        self.traces_ = self._build_traces()

    def _build_movie_template(self) -> xr.DataArray:
        return xr.DataArray(
            np.zeros((self.n_frames, self.frame_dims.height, self.frame_dims.width)),
            dims=[AXIS.frames_dim, *AXIS.spatial_dims],
        ).astype(np.float32)

    def _generate_footprint(
        self, radius: int, position: Position, id_: str, confidence: float
    ) -> xr.DataArray:
        footprint = xr.DataArray(
            np.zeros((self.frame_dims.height, self.frame_dims.width)),
            dims=AXIS.spatial_dims,
        )

        shape = disk(radius).astype(np.float32)

        width_slice = slice(position.width - radius, position.width + radius + 1)
        height_slice = slice(position.height - radius, position.height + radius + 1)

        footprint.loc[{"height": height_slice, "width": width_slice}] = shape

        return footprint.expand_dims(AXIS.component_dim).assign_coords(
            {
                AXIS.id_coord: (AXIS.component_dim, [id_]),
                AXIS.confidence_coord: (AXIS.component_dim, [confidence]),
                **{ax: footprint[ax] for ax in AXIS.spatial_dims},
            }
        )

    def _build_footprints(self) -> xr.DataArray:
        footprints = []
        for radius, position, id_, confid in zip(
            self.cell_radii, self.cell_positions, self.cell_ids, self.confidences
        ):
            footprints.append(self._generate_footprint(radius, position, id_, confid))

        return xr.concat(footprints, dim=AXIS.component_dim)

    def _format_trace(self, trace: np.ndarray, id_: str, confidence: float) -> xr.DataArray:
        return (
            xr.DataArray(
                trace,
                dims=AXIS.frames_dim,
            )
            .expand_dims(AXIS.component_dim)
            .assign_coords(
                {
                    AXIS.id_coord: (AXIS.component_dim, [id_]),
                    AXIS.confidence_coord: (AXIS.component_dim, [confidence]),
                    AXIS.frames_dim: range(trace.size),
                    AXIS.timestamp_coord: (
                        AXIS.frames_dim,
                        [
                            (datetime.now() + i * timedelta(microseconds=20)).strftime(
                                "%H:%M:%S.%f"
                            )
                            for i in range(trace.size)
                        ],
                    ),
                }
            )
        )

    def _build_traces(self) -> xr.DataArray:
        traces = []
        for trace, id_, confid in zip(self.cell_traces, self.cell_ids, self.confidences):
            traces.append(self._format_trace(trace, id_, confid))

        return xr.concat(traces, dim=AXIS.component_dim)

    def _build_movie(self, footprints: xr.DataArray, traces: xr.DataArray) -> xr.DataArray:
        movie = self._build_movie_template()
        movie += (footprints @ traces).transpose(AXIS.frames_dim, *AXIS.spatial_dims)
        return movie

    def make_movie(self) -> xr.DataArray:
        movie = self._build_movie(self.footprints_, self.traces_)
        return movie

    def add_cell(
        self, position: Position, radius: int, trace: np.ndarray, id_: str, confidence: float = 0.0
    ) -> None:
        new_footprint = self._generate_footprint(radius, position, id_, confidence)
        self.footprints_ = xr.concat([self.footprints_, new_footprint], dim=AXIS.component_dim)

        new_trace = self._format_trace(trace, id_, confidence)
        self.traces_ = xr.concat([self.traces_, new_trace], dim=AXIS.component_dim)

    def drop_cell(self, id_: str | Iterable[str]) -> None:
        id_ = {id_} if isinstance(id_, str) else set(id_)

        id_coords = set(self.footprints_.coords["id_"].values.tolist())

        keep_ids = list(id_coords - id_)

        self.footprints_ = (
            self.footprints_.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )
        self.traces_ = (
            self.traces_.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )

    @property
    def footprints(self) -> xr.DataArray:
        if self.footprints_.any():
            return self.footprints_
        else:
            raise ValueError("No footprints available")

    @property
    def traces(self) -> xr.DataArray:
        if self.traces_.any():
            return self.traces_
        else:
            raise ValueError("No traces available")
