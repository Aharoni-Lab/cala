from dataclasses import dataclass, field
from typing import Self

import xarray as xr
from river.base import SupervisedTransformer

from cala.streaming.core import Parameters, TransformerMeta, Traces, Footprints
from cala.streaming.stores.odl import Residual


@dataclass
class ResidualInitializerParams(Parameters):
    """Parameters for residual computation"""

    component_axis: str = "components"
    """Axis for components"""
    id_coordinates: str = "id_"
    type_coordinates: str = "type_"
    frames_axis: str = "frame"
    """Frames axis"""
    spatial_axes: tuple = ("height", "width")
    """Spatial axes for pixel statistics"""
    buffer_length: int = 50
    """Number of frames to keep in the residual buffer (l_b)"""

    def validate(self):
        if not isinstance(self.spatial_axes, tuple) or len(self.spatial_axes) != 2:
            raise ValueError("spatial_axes must be a tuple of length 2")
        if self.buffer_length <= 0:
            raise ValueError("buffer_length must be positive")


@dataclass
class ResidualInitializer(SupervisedTransformer, metaclass=TransformerMeta):
    """Computes residual buffer that contains the last l_b instances of the residual signal rt = yt − Act − bft,
    where l_b is a reasonably small number.

    Implements the equation: R_buf = [Y − [A, b][C; f]][:, t′ − l_b + 1 : t′]
    where:
    - Y is the data matrix (pixels x time)
    - [A, b] is the footprint matrix of both neuron and background
    - [C; f] is the traces matrix of both neuron and background
    - t' is the current timestep
    - R_buf is the resulting buffer
    """

    params: ResidualInitializerParams
    """Parameters for residual computation"""
    residual_: xr.DataArray = field(init=False)
    """Computed residual"""

    def learn_one(
        self, footprints: Footprints, traces: Traces, frame: xr.DataArray
    ) -> Self:
        """Learn residual from frames, temporal components, and footprints.

        Args:
            footprints: xarray DataArray of shape (components, height, width) containing spatial footprints
            traces: traces of all detected fluorescent components
            frame: xarray DataArray of shape (frames, height, width) containing 2D frames

        Returns:
            self
        """
        # Get current timestep
        t_prime = frame.sizes[self.params.frames_axis]

        # Reshape frames to pixels x time
        Y = frame.values.reshape(-1, t_prime)

        # Get temporal components [C; f]
        C = traces.values  # components x time

        # Reshape footprints to (pixels x components)
        A = footprints.values.reshape(
            footprints.sizes[self.params.component_axis], -1
        ).T

        # Compute residual R = Y - [A,b][C;f]
        R = Y - A @ C

        # Only keep the last l_b frames
        start_idx = max(0, t_prime - self.params.buffer_length)
        R = R[:, start_idx:]

        # Create xarray DataArray with proper dimensions and coordinates
        self.residual_ = xr.DataArray(
            R.reshape(*[frame.sizes[ax] for ax in self.params.spatial_axes], -1),
            dims=(*self.params.spatial_axes, self.params.frames_axis),
        )

        return self

    def transform_one(self, _=None) -> Residual:
        """Transform method returns the computed residual buffer.

        Returns:
            Residual buffer
        """
        return Residual(self.residual_)
