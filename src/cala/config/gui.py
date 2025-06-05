from pydantic import BaseModel


class MovieConfig(BaseModel):
    stream_url: str


class PlotConfig(BaseModel):
    width: int | str
    height: int | str
    max_points: int


class GUIConfig(BaseModel):
    prep_movie: MovieConfig
    metric_plot: PlotConfig
    footprint_movie: MovieConfig
