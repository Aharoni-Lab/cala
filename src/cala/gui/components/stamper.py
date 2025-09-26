from typing import Literal

import cv2
import numpy as np
import xarray as xr
from pydantic import BaseModel, ConfigDict

from cala.assets import Footprints, PopSnap
from cala.gui.components import Encoder
from cala.models import AXIS

COLOR_MAP = {
    "red": (0, 0, 1),
    "green": (0, 1, 0),
    "blue": (1, 0, 0),
    "yellow": (0, 1, 1),  # Green + Red
    "cyan": (1, 1, 0),  # Blue + Green
    "magenta": (1, 0, 1),  # Blue + Red
    "orange": (0, 0.65, 1),  # A common, approximate BGR for orange
    "purple": (0.5, 0, 0.5),  # A common, approximate BGR for purple
}


class DuckFrame(BaseModel):
    array_: xr.DataArray

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def array(self) -> xr.DataArray:
        return self.array_

    @array.setter
    def array(self, array: xr.DataArray) -> None:
        self.array_ = array

    @classmethod
    def from_array(cls, array: xr.DataArray) -> "DuckFrame":
        return cls(array_=array)


class Stamper(Encoder):
    color: Literal["gray", "rgb24"] = "rgb24"
    gain: float = 2.0

    def process(self, footprints: Footprints, new_traces: PopSnap) -> None:
        if footprints.array is None:
            return None

        frame = stamp(footprints, new_traces, gain=self.gain)
        super().process(frame=frame)

        return None


def stamp(footprints: Footprints, new_traces: PopSnap, gain: float) -> DuckFrame:
    """
    each footprint has a different color. (with intensity)
    categorize footprints into 5 buckets
    each bucket gets summed
    each bucket takes a color
    each bucket gets multiplied by the latest trace

    cv2.applyColorMap(gray_image, cv2.COLORMAP_HSV)
    """
    A = footprints.array * new_traces.array
    frame = np.zeros_like(
        cv2.cvtColor(
            A.isel({AXIS.component_dim: 0}).to_numpy().astype(np.uint8), cv2.COLOR_BGR2RGB
        ),
        dtype=int,
    )

    for i, (color, code) in enumerate(COLOR_MAP.items()):
        if i == A.sizes[AXIS.component_dim]:
            break
        partial = (
            cv2.cvtColor(
                A.isel(
                    {
                        AXIS.component_dim: list(
                            j for j in range(A.sizes[AXIS.component_dim]) if j % len(COLOR_MAP) == i
                        )
                    }
                )
                .sum(dim=AXIS.component_dim)
                .to_numpy()
                .astype(np.uint8),
                cv2.COLOR_GRAY2BGR,
            )
            * code
        )

        # since it centers to 128 before glow removal, this is probably safe
        frame += np.clip((partial * gain).astype(int), None, 255)

    return DuckFrame.from_array(xr.DataArray(frame, dims=[*AXIS.spatial_dims, "channel"]))
