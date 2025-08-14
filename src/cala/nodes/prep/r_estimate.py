from typing import Annotated as A
from typing import Any

import numpy as np
from noob import Name, process_method
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from skimage.feature import blob_log

from cala.assets import Frame
from cala.models import AXIS


class SizeEst(BaseModel):
    hardset_radius: int | None = None
    """if this is set, no learning occurs."""
    n_frames: int | None = None
    """how many first n frames to learn from. if none, keep learning forever"""
    log_kwargs: dict[str, Any] = Field(default_factory=dict)

    sizes_: list[float] = Field(default_factory=list)
    centers_: list[np.ndarray] = Field(default_factory=list)
    _est_radius: int = PrivateAttr(None)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @process_method
    def get_median_radius(self, frame: Frame) -> A[int, Name("radius")]:
        if self.hardset_radius:
            return self.hardset_radius

        if self.n_frames and self.n_frames < frame.array[AXIS.frame_coord]:
            return self._est_radius

        blobs = blob_log(frame.array, **self.log_kwargs)
        self.centers_ = [blobs[:-1] for blobs in blobs]
        self.sizes_ += [blob[-1].item() for blob in blobs]
        self._est_radius = int(np.round(np.median(self.sizes_)).item())

        return self._est_radius
