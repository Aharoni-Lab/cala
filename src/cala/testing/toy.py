from collections.abc import Generator, Iterable
from datetime import datetime, timedelta

import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from skimage.morphology import disk

from cala.assets import Footprints, Frame, Movie, Traces
from cala.models.axis import AXIS


class FrameDims(BaseModel):
    width: int
    height: int


class Position(BaseModel):
    width: int
    height: int


class Toy(BaseModel):
    """
    Ex:

    frame_dims = FrameSize(width=1024, height=512)
    cell_positions = [Position(width=200, height=300)]
    cell_traces = [np.array(range(100))]

    toy = Toy(
        n_frames=100,
        frame_dims=frame_dims,
        cell_radii=20,
        cell_positions=cell_positions,
        cell_traces=cell_traces,
    )

    cell_position = Position(width=400, height=300)
    cell_trace = np.array(range(100, 0, -1))

    toy.add_cell(cell_position, 10, cell_trace, "cell_a")

    toy.drop_cell("cell_a")

    toy_movie = toy.make_movie()
    """

    n_frames: int
    frame_dims: FrameDims
    cell_radii: int | list[int]
    cell_positions: list[Position]
    cell_traces: list[np.ndarray]
    cell_ids: list[str] | None = None
    """If none, auto populated as cell_{idx}."""
    confidences: list[float] = Field(default_factory=list)

    _footprints: xr.DataArray = PrivateAttr(init=False)
    _traces: xr.DataArray = PrivateAttr(init=False)

    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=True)

    def model_post_init(self, __context: None = None) -> None:
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

        self._footprints = self._build_footprints()
        self._traces = self._build_traces()

    def _build_movie_template(self) -> xr.DataArray:
        return xr.DataArray(
            np.zeros((self.n_frames, self.frame_dims.height, self.frame_dims.width)),
            dims=[AXIS.frames_dim, *AXIS.spatial_dims],
        )

    def _generate_footprint(
        self, radius: int, position: Position, id_: str, confidence: float
    ) -> xr.DataArray:
        footprint = xr.DataArray(
            np.zeros((self.frame_dims.height, self.frame_dims.width)),
            dims=AXIS.spatial_dims,
        )

        shape = disk(radius)

        width_slice = slice(position.width - radius, position.width + radius + 1)
        height_slice = slice(position.height - radius, position.height + radius + 1)

        footprint.loc[{AXIS.height_dim: height_slice, AXIS.width_dim: width_slice}] = shape

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
                }
            )
        )

    def _build_traces(self) -> xr.DataArray:
        traces = []
        for trace, id_, confid in zip(self.cell_traces, self.cell_ids, self.confidences):
            traces.append(self._format_trace(trace, id_, confid))

        return xr.concat(traces, dim=AXIS.component_dim).assign_coords(
            {
                AXIS.timestamp_coord: (
                    AXIS.frames_dim,
                    [
                        (datetime.now() + i * timedelta(microseconds=20)).strftime("%H:%M:%S.%f")
                        for i in range(self.n_frames)
                    ],
                )
            }
        )

    def _build_movie(self, footprints: xr.DataArray, traces: xr.DataArray) -> xr.DataArray:
        movie = self._build_movie_template()
        movie += (footprints @ traces).transpose(AXIS.frames_dim, *AXIS.spatial_dims)
        return movie

    def make_movie(self) -> Movie:
        movie = self._build_movie(self._footprints, self._traces)
        return Movie.from_array(movie)

    def add_cell(
        self, position: Position, radius: int, trace: np.ndarray, id_: str, confidence: float = 0.0
    ) -> None:
        new_footprint = self._generate_footprint(radius, position, id_, confidence)
        self._footprints = xr.concat([self._footprints, new_footprint], dim=AXIS.component_dim)

        new_trace = self._format_trace(trace, id_, confidence)
        self._traces = xr.concat([self._traces, new_trace], dim=AXIS.component_dim)

    def drop_cell(self, id_: str | Iterable[str]) -> None:
        id_ = {id_} if isinstance(id_, str) else set(id_)

        id_coords = set(self._footprints.coords[AXIS.id_coord].values.tolist())

        keep_ids = list(id_coords - id_)

        self._footprints = (
            self._footprints.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )
        self._traces = (
            self._traces.set_xindex(AXIS.id_coord)
            .sel({AXIS.id_coord: keep_ids})
            .reset_index(AXIS.id_coord)
        )

    @property
    def footprints(self) -> Footprints:
        if self._footprints.any():
            return Footprints.from_array(self._footprints)
        else:
            raise ValueError("No footprints available")

    @property
    def traces(self) -> Traces:
        if self._traces.any():
            return Traces.from_array(self._traces)
        else:
            raise ValueError("No traces available")

    def movie_gen(self) -> Generator[Frame]:
        for i in range(self._traces.sizes[AXIS.frames_dim]):
            trace = self._traces.isel({AXIS.frames_dim: i})
            yield Frame.from_array(trace @ self._footprints)
