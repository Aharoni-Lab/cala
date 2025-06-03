from pydantic import BaseModel


class MovieConfig(BaseModel):
    width: int
    height: int
    stream_url: str


class PlotConfig(BaseModel):
    width: int
    height: int
    max_points: int


class GUIConfig(BaseModel):
    prep_movie: MovieConfig
    metric_plot: PlotConfig
