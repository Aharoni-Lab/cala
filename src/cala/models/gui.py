from pydantic import BaseModel


class MovieSpec(BaseModel):
    stream_url: str


class PlotSpec(BaseModel):
    width: int | str
    height: int | str
    max_points: int


class GUISpec(BaseModel):
    prep_movie: MovieSpec
    metric_plot: PlotSpec
    footprint_movie: MovieSpec
