from pydantic import BaseModel


class MovieConfig(BaseModel):
    height: int
    width: int
    stream_url: str


class GUIConfig(BaseModel):
    raw_movie: MovieConfig
